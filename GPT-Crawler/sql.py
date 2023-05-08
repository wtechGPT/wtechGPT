import pymysql
import os
import time
import re
from selenium import webdriver
from urllib.parse import quote_plus

#####한글깨짐 방지###### 
os.environ["NLS_LANG"] = ".AL32UTF8"

# DB와 연결된 코드
conn = pymysql.connect(host = '152.67.200.79', user = 'yujin', password = 'Wlsehf0014@', db = 'gptCrawler', charset = 'utf8mb4', use_unicode=True)

db = conn.cursor()

def select_search(): # 질문 키워드 조회
    
    sql_select = 'select topic from quest_topic'
    db.execute(sql_select)
    quest_topic = db.fetchall()
    return quest_topic


def insert_search(data): # 크롤링 데이터 저장
    sql_insert = 'insert into topic_content (topic_title, topic_content, topic_reply_main, topic_reply_re) values (%s, %s, %s, %s)'
    val = (data[0], data[1], data[2], data[3])
    
    db.execute(sql_insert, val)
    conn.commit()
    print("DB저장 성공")

