import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pymysql
from scipy.stats import norm

# 날짜별================================================================================================================================
connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
                          charset='utf8')
cur = connect.cursor()
query = "WITH latest_activities AS (SELECT * FROM business_location_data WHERE si = '서울특별시') SELECT  a.date,a.time,a.category,a.location, IFNULL( VALUE, 0 )AS VALUE FROM business_sub_table AS a LEFT JOIN latest_activities AS k ON  (a.category =k.category AND k.dong = a.location AND a.time = k.time AND  a.date = k.date  )   WHERE (a.location = '구로구' OR a.location = '영등포구' OR a.location = '금천구' OR a.location = '도봉구' OR a.location = '은평구' ) AND (a.category = '패스트푸드' OR a.category = '분식' OR a.category = '치킨' OR a.category = '한식' OR a.category = '카페/디저트') AND (a.time = '0' OR a.time = '1' OR a.time = '10' OR a.time = '11' OR a.time = '12' OR a.time = '13' OR a.time = '14' OR a.time = '15' OR a.time = '16' OR a.time = '17' OR a.time = '18' OR a.time = '19' OR a.time = '20' OR a.time = '21' OR a.time = '22' OR a.time = '23') ORDER BY DATE, TIME ASC"

cur.execute(query)
select = np.array(cur.fetchall())
connect.commit()

x = select[:,0]
y = select[:,-1]
print(y.shape)
y = list(map(int, y))

for a in range(172800):
    y[a] = y[a] + 25
# sum(int(a+25) for a in list)
# plt.plot(x, y)
# plt.title('Date')
# plt.show()

# 정규분포
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic" # 한글 변환
plt.figure(figsize=(10,10))
sns.distplot(y, rug=True,fit=norm) #distplot 히스토그램 정규분포
plt.title("주문량 분포도",size=15, weight='bold')
plt.show()

y=np.sqrt(y)
plt.figure(figsize=(12,10))
sns.distplot(y, rug=True,fit=norm) #distplot 히스토그램 정규분포
plt.title("주문량 로그분포도",size=15, weight='bold')
plt.show()

# 이상치 확인
import numpy as np
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25,50,75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))
outlier_loc = outliers(y)
print("이상치의 위치 : ",outlier_loc)

# Q-Q plot & boxplot
from scipy.stats import norm
from scipy import stats
fig = plt.figure(figsize=(14,14))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
stats.probplot(y, plot=plt) 
green_diamond = dict(markerfacecolor='g', marker='D')
ax1.boxplot(y, flierprops=green_diamond)
plt.show()


# def outlier_z_score(data):
#     threshold = 2
#     mean = np.mean(data)
#     std = np.std(data)
#     z_scores = [(y-mean)/std for y in data]
#     return np.where(np.abs(z_scores)>threshold)

# print(outlier_z_score(y))
# for i in outlier_z_score(y):
#     print(y[i:1])
# # 시간별================================================================================================================================
# connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
#                           charset='utf8')
# cur = connect.cursor()
# query = "SELECT TIME,SUM(VALUE) FROM `time_location_data` WHERE DATE >= '2019-08-01' AND si = '서울특별시' GROUP BY TIME ORDER BY TIME ASC"
# cur.execute(query)
# select = np.array(cur.fetchall())
# connect.commit()

# x = select[:,0]
# y = select[:,1]

# plt.plot(x, y)
# plt.title('Time')
# plt.show()

# # 월별================================================================================================================================
# connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
#                           charset='utf8')
# cur = connect.cursor()
# query = "SELECT DATE_FORMAT(DATE,'%Y-%m'), SUM(VALUE) FROM `time_location_data` WHERE DATE >= '2019-08-01' AND si='서울특별시' GROUP BY DATE_FORMAT(DATE,'%Y-%m');"
# cur.execute(query)
# select = np.array(cur.fetchall())
# connect.commit()
# print(select)

# x = select[:,0]
# y = select[:,1]

# plt.plot(x, y)
# plt.title('Month')
# plt.show()

# # 요일별================================================================================================================================
# connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
#                           charset='utf8')
# cur = connect.cursor()
# query = "SELECT k.week, SUM(VALUE) FROM (SELECT * ,CASE DAYOFWEEK(DATE) WHEN '1' THEN 'sun' WHEN '2' THEN 'mon' WHEN '3' THEN 'tue' WHEN '4' THEN 'wed' WHEN '5' THEN 'thu' WHEN '6' THEN 'fri' WHEN '7' THEN 'sat' END AS WEEK FROM time_location_data WHERE DATE >= '2019-08-01' AND si = '서울특별시') AS k GROUP BY k.week"

# cur.execute(query)
# select = np.array(cur.fetchall())
# connect.commit()

# x = select[:,0]
# y = select[:,1]

# plt.plot(x, y)
# plt.title('Day')
# plt.show()

# # 동별================================================================================================================================
# connect = pymysql.connect(host='mitzy.c7xaixb8f0ch.ap-northeast-2.rds.amazonaws.com', user='mitzy', password='mitzy1234!', db='mitzy',\
#                           charset='utf8')
# cur = connect.cursor()
# query = "SELECT DATE_FORMAT(DATE,'%Y-%m') AS d_v,dong, SUM(VALUE)  AS m_v FROM `business_location_data` WHERE si = '서울특별시' GROUP BY DATE_FORMAT(DATE,'%Y-%m'),dong ORDER BY d_v, m_v DESC"
# cur.execute(query)
# select = np.array(cur.fetchall())
# connect.commit()

# x = select[:,0]
# y = select[:,1]

# plt.plot(x, y)
# plt.title('Dong')
# plt.show()