import requests
from selenium import webdriver
import time
from sql import select_search, insert_search
from selenium.webdriver.common.by import By


url = "https://wtech.inswave.kr/websquare/websquare.html?w2xPath=/ws/index.xml&inPath=/ws/member/login.xml"

driver = webdriver.Chrome('/Users/beomi/Downloads/chromedriver')
driver.implicitly_wait(3)

def scrape_login(): # w-tech 로그인 구현
    
    driver.get(url)
    time.sleep(2) # time.sleep 주지 않으면 아이디 입력이 잘 안됨.
    driver.find_element(By.CSS_SELECTOR, '#wf_frame_inputUserId').send_keys('lyujin@inswave.com')
    time.sleep(2)
    driver.find_element(By.CSS_SELECTOR, '#wf_frame_inputPassWord').send_keys('Wlsehf0014@')
    time.sleep(1)
    driver.find_element(By.CSS_SELECTOR, '#wf_frame_btnLogin').click()
    
    detail_search()
    

def detail_search(): # 검색 키워드 DB 조회
    searchText = select_search()
    
    for detail in searchText:
        detail_page(detail[0])    



def detail_page(searchText): # 각 키워드 별 데이터 크롤링

    print(searchText)

    for i in range(1, 21): # 각 키워드 검색시 나오는 페이지수 20개로 한정

        URL = "https://wtech.inswave.kr/websquare/websquare.html?w2xPath=/ws/index.xml&inPath=/ws/qna/qna_list.xml&searchType=title&searchText=" + searchText + "&curPage=" + str(i)
        driver.get(f"{URL}")
        
        print("################ URL : " + str(i) + " #######################")
        print(URL)
        
        for j in range(4, 10): # 한 페이지별 공지 4개 포함 10개 이므로 4 ~ 9 총 6개 데이터 추출
            
            img_name = '#wf_frame_grid1_cell_' + str(j) + '_1 > img'
            img = driver.find_element(By.CSS_SELECTOR, img_name)

            if img == '':
                continue

            else :
                img = img.get_attribute('src') # img 변수 재활용
                blue = "https://wtech.inswave.kr/images/ico_grd_blu.gif"
                
                if img != blue: # "해결" 인 질문 만 가져오게 설정
                    continue

                else :
                
                    detail_name = '#wf_frame_grid1_cell_' + str(j) + '_2 > nobr > a';
                    data = []
                    
                    title = driver.find_element(By.CSS_SELECTOR, detail_name).text # 질문 제목(title) 추출
                    print(title)

                    data.append(title) # title 저장

                    driver.find_element(By.CSS_SELECTOR, detail_name).click() # 각 질문 클릭해 detail 페이지 접속
                                
                    content = driver.find_element(By.CSS_SELECTOR, '#wf_frame_txtContent').text # 질문 내용 (content) 추출
                    
                    start = content.find('<< 개요 >>')
                    last = content.find('<< 버전 및 빌드일 >>')
                    final_content = content[start:last] # 질문 내용 중 필요부분 최종 str
                    
                    data.append(final_content) # final_content 저장
                    
                    reply_main_num = 0
                    reply_main_name = "#wf_frame_generator1_" + str(reply_main_num) + "_content"

                    final_reply_main = "" # 답변 depth = 0
                    final_reply_re = "" # 답변 depth > 0
                    
                    while(True): # reply_main_name 없을 때 까지 > reply_main_name 없다는건 답변(depth = 0)이 더이상 없다는 것.
                        try:
                            reply_main = driver.find_element(By.CSS_SELECTOR, reply_main_name).text # 답변 depth = 0 추출
                            
                            check = reply_main.find('?') # 답변 depth = 0 중 질문형식인 '?' 가 있는치 체크 > 대답 컬럼엔 질문이 들어가면 안되기 때문.
                            if check == -1:
                                temp = final_reply_main
                                final_reply_main = temp + reply_main + "\n\n" 
                            
                            reply_re_num = 0
                            reply_re_name = "#wf_frame_generator1_" + str(reply_main_num) + "_generator2_" + str(reply_re_num) + "_content" # 답변 depth > 0 추출
                            
                            while(True): # reply_re_name 없을 때 까지 > reply_main_name 없다는건 답변(depth > 0)이 더이상 없다는 것.
                                try:
                                    reply_re = driver.find_element(By.CSS_SELECTOR, reply_re_name).text
                                    
                                    check = reply_re.find('?') # 답변 depth > 0 중 질문형식인 '?' 가 있는치 체크 > 대답 컬럼엔 질문이 들어가면 안되기 때문.
                                    
                                    if check == -1:
                                        temp = final_reply_re
                                        final_reply_re = temp + reply_re + "\n\n" 
                                        
                                    reply_re_num += 1
                                    reply_re_name = "#wf_frame_generator1_" + str(reply_main_num) + "_generator2_" + str(reply_re_num) + "_content"
                                except:
                                    break
                                
                            reply_main_num += 1
                            reply_main_name = "#wf_frame_generator1_" + str(reply_main_num) + "_content"
                            

                        except:
                            break

                    # 최종 답변 저장    
                    data.append(final_reply_main)
                    data.append(final_reply_re)
                    
                    insert_search(data) # 크롤링 데이터 db에 저장
                    
                    driver.back() # driver 종료
                
    
    
    