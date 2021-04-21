import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



"WITH latest_activities AS (SELECT * FROM business_location_data WHERE si = '서울특별시' )\
SELECT  a.date,a.time,a.category,a.location, IFNULL( VALUE, 0 )AS VALUE\
FROM business_sub_table AS a\
LEFT JOIN latest_activities AS k\
ON  (a.category =k.category AND k.dong = a.location AND a.time = k.time AND  a.date = k.date  ) WHERE (a.time = '0' OR a.time = '1' OR a.time = '10' OR a.time = '11' OR a.time = '12' OR a.time = '13' OR a.time = '14' OR a.time = '15' OR a.time = '16' OR a.time = '17' OR a.time = '18' OR a.time = '19' OR a.time = '20' OR a.time = '21' OR a.time = '22' OR a.time = '23')\
AND (a.category = '패스트푸드' OR a.category = '치킨' OR a.category = '분식' OR a.category = '한식' OR a.category = '카페/디저트' )\
ORDER BY DATE, TIME ASC"

import matplotlib

# 자료불러오기
df = pd.read_csv('C:\order_project\시간-지역별 배달 주문건수_20201031000000.csv',thousands = ',', index_col=0, header=None)
print(df.head())

is_seoul = df[2] == '서울특별시'
Seoul = df[is_seoul]
print(Seoul.head())
# Seoul.to_csv('C:/order_project/seoul_data.csv',index=True)

# 날짜별
Seoul = Seoul.drop(Seoul.columns[[0,1,2,3]], axis='columns')
print(Seoul.head())
date_sum = Seoul.groupby(0).sum()
print(date_sum.head())

# # 시간별
# date_sum = Seoul.groupby(1).sum()
# print(date_sum.head())

# plt.figure()
# ax = date_sum.plot.line()
# plt.show()

# 요일별
Seoul = Seoul.drop(Seoul.columns[[0,1,2,3]], axis='columns')
print(Seoul.head())
date_sum = Seoul.groupby(0).sum()
print(date_sum.head())
date_sum = date_sum.values

x0 = ['wed','thu','fri','sat','sun','mon','tue']
y0 = []
for a in range(7):
    b = date_sum[a::7,:].sum()
    y0.append(b)

plt.bar(x0, y0)
plt.show()

# 월별
# Seoul = Seoul.drop(Seoul.columns[[0,1,2,3]], axis='columns')
# date_sum=[]
# for m in range(1,13):
#     a = Seoul['2019-'+"%02d"%m+'-01':'2019-'+"%02d"%m+'-30'].values.sum()
#     date_sum.append(a)
# for m in range(1,13):
#     a = Seoul['2020-'+"%02d"%m+'-01':'2020-'+"%02d"%m+'-30'].values.sum()
#     date_sum.append(a)

# month = ['19-01','19-02','19-03','19-04','19-05','19-06','19-07','19-08','19-09','19-10','19-11','19-12',
#     '20-01','20-02','20-03','20-04','20-05','20-06','20-07','20-08','20-09','20-10','20-11','20-12']

# plt.bar(month, date_sum)
# plt.show()
