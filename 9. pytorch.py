import db_connect as db
import numpy as np
import pandas as pd
import itertools
from IPython.display import Image
from IPython import display
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data_utils 

def load_data(query, is_train = True):
    query = query
    db.cur.execute(query)
    dataset = np.array(db.cur.fetchall())

    # pandas 넣기
    column_name = ['date', 'year', 'month', 'day', 'time', 'category', 'dong','temperature','rain','wind','humidity','person', 'value']
    df = pd.DataFrame(dataset, columns=column_name)
    db.connect.commit()

    if is_train == True:
        # train, test 나누기
        train_value = df[ '2020-09-01' > df['date'] ]
        x = train_value.iloc[:,1:-1]
        y = train_value.iloc[:,-1]
    else:
        test_value = df[df['date'] >=  '2020-09-01']
        x = test_value.iloc[:,1:-1]
        y = test_value.iloc[:,-1]

    
    # 원 핫으로 컬럼 추가해주는 코드!!!!!    
    x = pd.get_dummies(x, columns=["category", "dong"])
    # 카테고리랑 동만 원핫으로 해준다 

    return x, y

x_pred_pd, y_pred_pd = load_data("SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity, IFNULL( c_person,0) AS person, VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time LEFT JOIN `covid19_re` AS c ON c.date = d.date ", is_train = False)
x_train_pd, y_train_pd = load_data("SELECT d.date,YEAR,MONTH, d.day, d.time, category, dong,temperature,rain,wind,humidity, IFNULL( c_person,0) AS person, VALUE FROM main_data_table AS d INNER JOIN `weather` AS s ON d.date = s.DATE AND d.time = s.time LEFT JOIN `covid19_re` AS c ON c.date = d.date WHERE (d.TIME != 2 AND d.TIME != 3 AND d.TIME != 4 AND d.TIME != 5  AND d.TIME != 6 AND d.TIME != 7 AND d.TIME != 8) ORDER BY DATE, YEAR, MONTH, DAY, TIME, category, dong ASC")

# 파이터치 넘파이로 변환 ,from_numpy로 만들어진 텐서는 해당 ndarray와 메모리를 공유
trn_X  = torch.from_numpy(x_train_pd.astype(float).values)
trn_y  = torch.from_numpy(y_train_pd.astype(float).values)

val_X  = torch.from_numpy(x_pred_pd.astype(float).values)
val_y  = torch.from_numpy(y_pred_pd.astype(float).values)

# for dictionary batch
class Dataset(data_utils.Dataset):

    #__init__()는 클래스의 생성자 역할
    def __init__(self, X, y):  
        self.X = X
        self.y = y
   
    # 인덱스로 접근할 수 있는 이터레이터
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'y': self.y[idx]}
   
    def __len__(self):
        return len(self.X)

batch_size=500

# pytorch dataloader
trn = Dataset(trn_X, trn_y)
trn_loader = data_utils.DataLoader(trn, batch_size=batch_size, shuffle=True)

val = Dataset(val_X, val_y)
val_loader = data_utils.DataLoader(val, batch_size=batch_size, shuffle=False)

tmp = next(iter(trn_loader)) # 1. iterator 생성,  2. next 값을 차례대로 꺼내기
# print(tmp)

print("Training Shape", trn_X.shape, trn_y.shape) # Training Shape torch.Size([2459424, 47]) torch.Size([2459424])
print("Testing Shape", val_X.shape, val_y.shape) # Testing Shape torch.Size([177408, 47]) torch.Size([177408])  

num_batches = len(trn_loader)

# Build Model
use_cuda = torch.cuda.is_available()