
import pymysql
import numpy as np

connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
                          charset='utf8')
cur = connect.cursor()


query = "SELECT * FROM test1"
cur.execute(query)
select = np.array(cur.fetchall())

connect.commit()

print(select.shape)   #튜플 형태로 반환받는다.
