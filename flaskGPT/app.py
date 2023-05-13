from flask import Flask, url_for , render_template, request ,jsonify
from markupsafe import escape
import openai
import numpy
import requests
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import tiktoken
import pandas as pd
import os
import matplotlib
from markupsafe import escape

app = Flask(__name__)

def page_not_found(e):
  return render_template('404.html'), 404

app.register_error_handler(404, page_not_found)


openai.organization = "org-dLiMqqSmLIXuV45cMCo7Pvcj"
# openai.api_key = os.getenv("OPENAI_API_KEY" )
openai.api_key = 자기 키 넣기
openai.Model.list()



''' 은별이 코드 시작 '''

df_n1apiKo1=pd.read_csv('processed/embeddings_ko.csv', index_col=0)
df_n1apiKo1['embeddings'] = df_n1apiKo1['embeddings'].apply(eval).apply(numpy.array)


def create_context_ko(
        question, df, max_len=1500, debug=False
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0
    prev_distance = 0
    prev_msg = ""

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        if prev_distance == row['distances']:
            #             print("이전 목록과 동일 (distance)")
            continue
        elif prev_msg == row['n1DataKo']:
            #             print("이전 목록과 동일 (문자열)")
            continue
        else:
            prev_distance = row['distances']
            prev_msg = row['n1DataKo']

            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 4

            # If the context is too long, break
            if cur_len > max_len:
                break

            if debug:
                print(i, row['distances'], row['n1DataKo'])
            # Else add it to the text that is being returned
            returns.append(row["n1DataKo"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question_chat_ko(
        df,
        model="gpt-3.5-turbo",
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=3000,
        debug=False,
        #     max_tokens=150,
        stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context_ko(
        question,
        df,
        max_len=max_len,
        debug=debug
    )
    # If debug, print the raw model response
    if debug:
        print("\n\nContext:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user",
                 "content": f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"잘 모르겠습니다.\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer: Video title - Show it in link embedding format and put in a new line character \n when you're done with one topic"}
            ],
            temperature=0,
            #             max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )

        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return ""


'''은별이 코드 끝 '''



'''지수언니 코드 시작 '''

tokenizer = tiktoken.get_encoding("cl100k_base")

df_n1apiKo=pd.read_csv('processed/topic_content_final.csv', index_col=0 , encoding='cp949')
# df_n1apiKo.columns = ['유형', 'component', 'name', 'description', 'parameter', 'return', 'exception', 'sample', 'built since', 'built last']
df_n1apiKo["text"] = df_n1apiKo['topic_title'].astype(str) +"    "+ df_n1apiKo["topic_content"].astype(str) +"    "+ df_n1apiKo["topic_reply_main"].astype(str) +"    "+ df_n1apiKo["topic_reply_re"].astype(str)


# df_n1apiKo["text"] = df_n1apiKo['text_1'].astype(str) + "    "

# tokenize the text and save the number of tokens to a new column
df_n1apiKo['n_tokens'] = df_n1apiKo.text.apply(lambda x: len(tokenizer.encode(str(x))))

max_tokens = 500


# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens=max_tokens):
    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    chunks.append(". ".join(chunk) + ".")
    return chunks


shortened_ko = []

# Loop through the dataframe
for row in df_n1apiKo.iterrows():

    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened_ko += split_into_many(row[1]['text'])

    # Otherwise, add the text to the list of shortened texts
    else:
        shortened_ko.append(row[1]['text'])



df_ko_a = pd.DataFrame(shortened_ko, columns = ['text'])
df_ko_a['n_tokens'] = df_n1apiKo.text.apply(lambda x: len(tokenizer.encode(x)))

max_tokens_cleaning = 2500

cleaning_ko = []

# Loop through the dataframe
for row in df_ko_a.iterrows():

    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens_cleaning:
        cleaning_ko += split_into_many(row[1]['text'], max_tokens=max_tokens_cleaning)

    # Otherwise, add the text to the list of shortened texts
    else:
        cleaning_ko.append(row[1]['text'])

df_cleaning_ko = pd.DataFrame(cleaning_ko, columns = ['text'])
df_cleaning_ko['n_tokens'] = df_cleaning_ko.text.apply(lambda x: len(tokenizer.encode(x)))



df_cleaning_ko['embeddings'] = df_cleaning_ko.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])


def create_context(
        question, df, max_len=1640, debug=False
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0
    prev_distance = 0
    prev_msg = ""

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        if prev_distance == row['distances']:
            #             print("이전 목록과 동일 (distance)")
            continue
        elif prev_msg == row['text']:
            #             print("이전 목록과 동일 (문자열)")
            continue
        else:
            prev_distance = row['distances']
            prev_msg = row['text']

            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 5

            # If the context is too long, break
            if cur_len > max_len:
                break

            if debug:
                print(i, row['distances'], row['text'])
            # Else add it to the text that is being returned
            returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question_chat(
        df,
        #     model="gpt-3.5-turbo",
        model="gpt-4",
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=3000,
        debug=False,
        #     max_tokens=150,
        stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        debug=debug
    )
    # If debug, print the raw model response
    if debug:
        print("\n\nContext:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user",
                 "content": f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"잘 모르겠습니다.\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:  Exclude sentences that contain the meaning of the words 'Attach sample file' and 'Enter project name'.. answer me in korean. Make your answer 50% the length of the original. You may use only the example code in the original text. Please exclude empty examples."}
            ],
            temperature=0,
            #             max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )

        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return ""





'''지수언니 코드 끝 '''




@app.route('/')
def index():
    return render_template('index.html')
    # return render_template('index.html', **locals())



@app.route('/youtube',  methods=['GET','POST'])
def chat():
    if request.method == 'POST' :


        value = request.form['prompt'].lower()
        print("prompt>>>>>>>>>" , value)


        isFirst =  request.form['isFirst']
        print("isFirst>>>", isFirst)



        if isFirst == 'False':
            find_list = [ '링크', '유튜브', 'youtube', '유투브']
            str_cnt = 0
            for i in find_list:
                str_cnt += value.count(i)

            res = {}

            if str_cnt > 0:
                print("youtube")
                res['answer'] = answer_question_chat_ko(df_n1apiKo1, question=value, debug=False)  # 은별이 코드
            else:
                print("Q&A")
                res['answer'] = answer_question_chat(df_cleaning_ko, question=value)  # 지수언니 코드

            print("str_cnt>>>", str_cnt)

            res['answer'] = escape(res['answer'] )
            print(">>>>>>answer>>>>", res['answer'])
            return jsonify(res) , 200



    return render_template('chat.html' , question=value)




if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

