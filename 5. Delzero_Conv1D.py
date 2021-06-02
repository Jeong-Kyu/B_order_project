import imp
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
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout, UpSampling2D, Conv2D, Conv1D

# db 직접 불러오기 


# 0 없다
'''
query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"
'''

# # 0 있다
# query = "SELECT * FROM main_data_table WHERE (TIME != 2 AND TIME != 3 AND TIME != 4 AND TIME != 5  AND TIME != 6 AND TIME != 7 AND TIME != 8) ORDER BY DATE, YEAR, MONTH ,TIME, category ASC  "
# query1 = "select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC"
# db.cur.execute(query)
# dataset = np.array(db.cur.fetchall())
# db.cur.execute(query1)
# dataset1 = np.array(db.cur.fetchall())

# # pandas 넣기

# column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

# df = pd.DataFrame(dataset, columns=column_name)
# df1 = pd.DataFrame(dataset1, columns=column_name)

# db.connect.commit()

# train_value = df[ '2020-09-01' > df['date'] ]

# x_train = train_value.iloc[:,1:-1].astype('int64')
# y_train = train_value['value'].astype('int64').to_numpy()

# test_value = df1[df1['date'] >=  '2020-09-01']

# x_pred = test_value.iloc[:,1:-1].astype('int64')
# y_pred = test_value['value'].astype('int64').to_numpy()

# x_train = pd.get_dummies(x_train, columns=["category", "dong"]).to_numpy()
# x_pred = pd.get_dummies(x_pred, columns=["category", "dong"]).to_numpy()


# np.save("C:/data/npy/p0_order_x_train.npy", arr=x_train)
# np.save("C:/data/npy/p0_order_y_train.npy", arr=y_train)
# np.save("C:/data/npy/p0_order_x_pred.npy", arr=x_pred)
# np.save("C:/data/npy/p0_order_y_pred.npy", arr=y_pred)

x_train = np.load("C:/data/npy/p0_order_x_train.npy",allow_pickle=True)
y_train = np.load("C:/data/npy/p0_order_y_train.npy",allow_pickle=True)
x_pred = np.load("C:/data/npy/p0_order_x_pred.npy",allow_pickle=True)
y_pred = np.load("C:/data/npy/p0_order_y_pred.npy",allow_pickle=True)


def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.9, random_state = 77, shuffle=True ) 
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
x_pred = x_pred.reshape(x_pred.shape[0],x_pred.shape[1],1)


print(x_train.shape, x_val.shape, x_pred.shape) # (3124915, 42) (347213, 42) (177408, 42)


inputs = Input(shape=(x_train.shape[1],1),name='input')
x = Conv1D(filters = 1024, kernel_size = 2, padding = 'same', strides = 2, activation='selu')(inputs)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(256,activation='selu')(x)
x = Dropout(0.2)(x)
x = Dense(64,activation='selu')(x)
x = Dense(16,activation='selu')(x)
outputs = Dense(1)(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()


es= EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
cp = ModelCheckpoint('../data/h5/p0_conv1_1.hdf5', monitor='val_loss', save_best_only=True, verbose=1,mode='auto')
model.compile(loss='mse', optimizer='RMSprop', metrics='mae')
model.fit(x_train, y_train, epochs=50, batch_size=2048, validation_data=(x_val,y_val), callbacks=[reduce_lr,cp] )

# from tensorflow.keras.models import load_model
# model = load_model('../data/h5/p0_dense_1.hdf5', custom_objects={'leaky_relu': tf.nn.leaky_relu})

# 4. 평가, 예측

loss, mae = model.evaluate(x_pred, y_pred, batch_size=1024)
y_predict = model.predict(x_pred)
print('loss : ',loss)
# RMSE 
print("RMSE : ", RMSE(y_pred, y_predict))

# R2 만드는 법
r2 = r2_score(y_pred, y_predict)
print("R2 : ", r2)


import matplotlib.pyplot as plt
 
fig = plt.figure( figsize = (12, 4))
chart = fig.add_subplot(1,1,1)
chart.plot(y_pred, marker='o', color='blue', label='실제값')
chart.plot(y_predict, marker='^', color='red', label='예측값')
plt.legend(loc = 'best') 
plt.show()