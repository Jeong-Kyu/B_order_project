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
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation, Dropout, UpSampling2D, Conv2D, ZeroPadding2D, MaxPooling2D, Add
from tensorflow.keras.applications.efficientnet import preprocess_input

# db 직접 불러오기 

# 0 있다
query = "SELECT * FROM main_data_table WHERE (TIME != 2 AND TIME != 3 AND TIME != 4 AND TIME != 5  AND TIME != 6 AND TIME != 7 AND TIME != 8) ORDER BY DATE, YEAR, MONTH ,TIME, category ASC  "
query1 = "select * from main_data_table ORDER BY DATE, YEAR, MONTH ,TIME, category ASC"
db.cur.execute(query)
dataset = np.array(db.cur.fetchall())
db.cur.execute(query1)
dataset1 = np.array(db.cur.fetchall())

# pandas 넣기

column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

df = pd.DataFrame(dataset, columns=column_name)
df1 = pd.DataFrame(dataset1, columns=column_name)

db.connect.commit()

train_value = df[ '2020-09-01' > df['date'] ]

x_train = train_value.iloc[:,1:-1].astype('int64')
y_train = train_value['value'].astype('int64').to_numpy()

test_value = df1[df1['date'] >=  '2020-09-01']

x_pred = test_value.iloc[:,1:-1].astype('int64')
y_pred = test_value['value'].astype('int64').to_numpy()

x_train = pd.get_dummies(x_train, columns=["category", "dong"]).to_numpy()
x_pred = pd.get_dummies(x_pred, columns=["category", "dong"]).to_numpy()

x_train = preprocess_input(x_train) # 
x_pred = preprocess_input(x_pred)   # 

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,  train_size=0.9, random_state = 77, shuffle=True ) 
# x_train = x_train.reshape(38833, 38, 28, 3)
print(x_train.shape, x_val.shape, x_pred.shape) # (3124915, 42) (347213, 42) (177408, 42)

x_train = x_train.reshape(x_train.shape[0], 42,1,1)
x_val = x_val.reshape(x_val.shape[0], 42,1,1)
x_pred = x_pred.reshape(x_pred.shape[0], 42,1,1)

resnet = ResNet50(include_top=False, input_shape=(42,42,3), weights='imagenet')
inputs = Input(shape=(x_train.shape[1],1,1),name='input')
a = Conv2D(3, kernel_size=(1,1))(inputs) # (None, 42, 1, 3)  
a = UpSampling2D(size=(1,42))(a) # 모양 맞추기 # (None, 42, 42, 3) 

def conv1_layer(x):    
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
 
    return x   
 
def conv2_layer(x):         
    x = MaxPooling2D((3, 3), 2)(x)     
 
    shortcut = x
 
    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)            
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  
 
            shortcut = x        
    
    return x
 
def conv3_layer(x):        
    shortcut = x    
    
    for i in range(4):     
        if(i == 0):            
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)    
 
            shortcut = x              
        
        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu')(x)
 
            shortcut = x      
            
    return x
 
def conv4_layer(x):
    shortcut = x        
  
    for i in range(6):     
        if(i == 0):            
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)
 
            shortcut = x               
        
        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)
 
            shortcut = x      
 
    return x
 
def conv5_layer(x):
    shortcut = x    
  
    for i in range(3):     
        if(i == 0):            
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu')(x)      
 
            shortcut = x               
        
        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)       
 
            shortcut = x                  
 
    return x
 
x = conv1_layer(a)
x = conv2_layer(x)
x = conv3_layer(x)
x = conv4_layer(x)
x = conv5_layer(x)
x = GlobalAveragePooling2D()(x) # (None, 2, 2, 2048)
x = Flatten()(x) # (None, 8192)
x = Dense(32, activation='relu')(x) # (None, 8)
# 히든 레이어를 최대한 많이 출력해보자
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

# # 3. 컴파일 훈련
es= EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
# cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
cp = ModelCheckpoint('../data/h5/resnet_dense_1.hdf5', monitor='val_loss', save_best_only=True, verbose=1,mode='auto')
model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=10, batch_size=512, validation_data=(x_val,y_val), callbacks=[es,reduce_lr,cp] )

# 4. 평가, 예측

loss, mae = model.evaluate(x_pred, y_pred, batch_size=64)
y_predict = model.predict(x_pred)

# RMSE 
print("RMSE : ", RMSE(y_pred, y_predict))

# R2 만드는 법
r2 = r2_score(y_pred, y_predict)
print("R2 : ", r2)

# 엑셀 추가 코드 
# 경로 변경 필요!!!!

df = pd.DataFrame(y_predict)
df['test'] = y_pred
df.to_csv('../data/team_project/resnet_dense.csv',index=False)

import matplotlib.pyplot as plt
 
fig = plt.figure( figsize = (12, 4))
chart = fig.add_subplot(1,1,1)
chart.plot(y_pred, marker='o', color='blue', label='실제값')
chart.plot(y_predict, marker='^', color='red', label='예측값')
plt.legend(loc = 'best') 
plt.show()