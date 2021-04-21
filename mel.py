import numpy as np
import db_connect as db
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# db 직접 불러오기 

size = 2688 #30

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))

def split_xy(dataset, time_steps, y_c):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end = i + time_steps
        y_end = x_end + y_c

        if y_end > len(dataset):
            break
        tmp_x = dataset[i:x_end, :-1]
        tmp_y = dataset[x_end:y_end, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

for main_num in range(1):

    '''
    query = "SELECT a.date, IF(DATE LIKE '2019-%', '2019', '2020') AS YEAR , CASE WHEN DATE LIKE '%-01-%' THEN '1' WHEN  DATE LIKE '%-02-%' THEN '2' WHEN  DATE LIKE '%-03-%' THEN '3' WHEN  DATE LIKE '%-04-%' THEN '4'\
    WHEN  DATE LIKE '%-05-%' THEN '5' WHEN  DATE LIKE '%-06-%' THEN '6' WHEN  DATE LIKE '%-07-%' THEN '7' WHEN  DATE LIKE '%-08-%' THEN '8' WHEN  DATE LIKE '%-09-%' THEN '9' WHEN  DATE LIKE '%-10-%' THEN '10' WHEN  DATE LIKE '%-11-%' THEN '11' ELSE '12' END AS MONTH,\
    DAYOFWEEK (DATE)AS DAY, a.time, s.index AS category, d.index AS dong, a.value FROM `business_location_data` AS a INNER JOIN `category_table` AS s ON a.category = s.category INNER JOIN `location_table` AS d ON a.dong = d.location WHERE si = '서울특별시'"
    '''
    query = 'SELECT * FROM main_data_table WHERE dong = "' + str(main_num) + '" ORDER BY DATE, YEAR, MONTH ,TIME, category ASC'

    db.cur.execute(query)
    dataset = np.array(db.cur.fetchall())

    # pandas 넣기

    column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong', 'value']

    df = pd.DataFrame(dataset, columns=column_name)

    db.connect.commit()

    # train, test 나누기

    train_value = df[ '2020-09-01' >= df['date'] ]
    train_value = train_value.iloc[:,1:].astype('int64').to_numpy()

    #x_train = train_value.iloc[:,1:-1].astype('int64').to_numpy()
    #y_train = train_value.iloc[:,-1].astype('int64').to_numpy()

    test_value = df[df['date'] >=  '2020-09-01']
    test_value = test_value.iloc[:,1:].astype('int64').to_numpy()

    #x_pred = test_value.iloc[:,1:-1].astype('int64').to_numpy()
    #y_pred = test_value.iloc[:,-1].astype('int64').to_numpy()

    kfold = KFold(n_splits=3, shuffle=True)
    
   
    x_train, y_train = split_xy(train_value, 3696, 3696)
    x_pred, y_pred = split_xy(test_value, 3696, 3696)
    print(x_train.shape)
    print(y_train.shape)
    print(x_train[0])
    print(x_train[1])
