import numpy as np
import db_connect as db
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB2, EfficientNetB7, VGG16, MobileNet, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout, UpSampling2D, Conv2D

# db 직접 불러오기 


# 0 없다
'''
query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"
'''

# 0 있다
query = "select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC"

db.cur.execute(query)
dataset = np.array(db.cur.fetchall())


# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)

db.connect.commit()

# train, test 나누기
# 원 핫으로 컬럼 추가해주는 코드!!!!!
df = pd.get_dummies(df, columns=["category", "dong"])
# 카테고리랑 동만 원핫으로 해준다 

# df.fit_transform(X, y)

# train, test 나누기

train_value = df[ '2020-09-01' > df['date'] ]

x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1,1)

test_value = df[df['date'] >=  '2020-09-01']

x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1],1,1)

# x_pred = x_pred.reshape(x_pred.shape[0], 32,32,3) # (2777702, 42) (694426, 42) (177408, 42)

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.9, random_state = 77, shuffle=True ) 
# x_train = x_train.reshape(38833, 38, 28, 3)
print(x_train.shape, x_val.shape, x_pred.shape) # (3124915, 42) (347213, 42) (177408, 42)

effinet = EfficientNetB2(include_top=False, input_shape=(42,42,3))
inputs = Input(shape=(x_train.shape[1],1,1),name='input')
a = Conv2D(3, kernel_size=(1,1))(inputs)
a = UpSampling2D(size=(1,42))(a)
x = effinet(a)
x = Flatten()(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()


# es= EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
# cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
cp = ModelCheckpoint('../data/h5/effiB2__dense_1.hdf5', monitor='val_loss', save_best_only=True, verbose=1,mode='auto')
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=10, batch_size=512, validation_data=(x_val,y_val), callbacks=[reduce_lr,cp] )

# 4. 평가, 예측

loss, mae = model.evaluate(x_pred, y_pred, batch_size=512)
y_predict = model.predict(x_pred)

# RMSE 
print("RMSE : ", RMSE(y_pred, y_predict))

# R2 만드는 법
r2 = r2_score(y_pred, y_predict)
print("R2 : ", r2)
