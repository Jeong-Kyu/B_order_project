import numpy as np
import db_connect as db
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# db 직접 불러오기 


# 0 없다
'''
query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"
'''

# 0 있다
query = "select * from main_data_table"


db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()

# train, test 나누기

train_value = df[ '2020-09-01' >= df['date'] ]

# x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
# y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

test_value = df[df['date'] >=  '2020-09-01']

# x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
# y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

def split_xy(dataset, time_steps, y_c):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end = i + time_steps
        y_end = x_end + y_c

        if y_end > len(dataset):
            break
        tmp_x = dataset[i:x_end,:-1]
        tmp_y = dataset[x_end:y_end, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x_train, y_train = split_xy(train_value, 352, 352)
x_pred, y_pred = split_xy(test_value, 352, 352)


# kfold = KFold(n_splits=5, shuffle=True)

# num = 0 


# r2_list = []
# rmse_list = []
# loss_list = []



# x_train1, x_test1 = x_train[train_index], x_train[test_index]
# y_train1, y_test1 = y_train[train_index], y_train[test_index]

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.8, random_state = 77, shuffle=True ) 

# 2. 모델구성

model = Sequential()
model.add(Dense(128, activation='relu' ,input_dim= 6)) 
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1)) 

# 3. 컴파일 훈련

modelpath = '../data/modelcheckpoint/team_.hdf5'
es= EarlyStopping(monitor='val_loss', patience=10)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val,y_val), callbacks=[es,reduce_lr] )

# 4. 평가, 예측

loss, mae = model.evaluate(x_test, y_test, batch_size=64)
y_predict = model.predict(x_pred)

# RMSE 
print("RMSE : ", RMSE(y_pred, y_predict))

# R2 만드는 법
r2 = r2_score(y_pred, y_predict)
print("R2 : ", r2)

# r2_list.append(r2_score(y_pred, y_predict))
# rmse_list.append(RMSE(y_pred, y_predict))
# loss_list.append(loss)

# num += 1


# print("r2 : ",r2_list)
# print("RMSE : ",rmse_list)
# print("loss : ",loss_list)



'''
from matplotlib import pyplot as plt
pred_test = model.predict(X_test)
plt.plot(pred_test,'r')
plt.plot(y_test,'g')
'''




# 0 있음  false
# r2 :  [0.29914470028609685, 0.4080221460750376, -0.0004918308414794126, 0.6231943856732425, 0.5481310630025362]
# RMSE :  [3.2494229398004624, 2.986378093796016, 3.882387094851552, 2.382597840303829, 2.6091455397497]
# loss :  [8.222070693969727, 7.295978546142578, 12.393566131591797, 4.843005657196045, 5.948018550872803]



# 0 있음  True
