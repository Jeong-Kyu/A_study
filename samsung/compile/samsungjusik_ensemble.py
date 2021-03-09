import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append(subset) 
    return np.array(aaa)

# 자료불러오기
df = pd.read_csv('./samsung/csv/0114삼성전자.csv',thousands = ',', encoding='UTF8', index_col=0, header=0)
df3 = pd.read_csv('./samsung/csv/삼성전자3.csv',thousands = ',', encoding='cp949', index_col=0, header=0)
df4 = pd.read_csv('./samsung/csv/KODEX 코스닥150 선물인버스.csv',thousands = ',', encoding='cp949', index_col=0, header=0)


# 자료 분할

sj2 = df3[['시가','고가','저가','종가','등락률','기관','프로그램']]
sj2 = sj2[:2]
sj = sj2.append(df[1:], ignore_index=True)
sj = sj.astype('float32')
sj = sj[::-1]
# print(sj.shape)
sj.to_csv('./samsung/csv/0115삼성전자.csv', encoding='utf-8-sig')

kod = df4[['시가','고가','저가','종가','신용비','개인','기관','외인(수량)']]
kod = kod[:664]
kod = kod.astype('float32')
kod = kod[::-1]
sj.to_csv('./samsung/csv/0115코스닥.csv', encoding='utf-8-sig')


# 상관계수를 통한 자료 선정
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df4.corr(), square=True, annot=True, cbar=True)
plt.show()
'''
# print(sj)

# 데이터
sj=sj.to_numpy()
kod=kod.to_numpy()

np.save('./samsung/save/data/sam_210120_sam_xy.npy', arr=sj)
np.save('./samsung/save/data/sam_210120_kod_xy.npy', arr=kod)

x=sj[:-2, :]
y=sj[1:, 0]
y = split_x(y,2)
# print(x.shape) #(662, 7)
# print(y.shape) #(662, 2)

x1=kod[:-2, :]
# print(x1.shape) #(662, 8)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=66, shuffle=True)

x1_train, x1_test = train_test_split(x1, train_size=0.8, random_state=66)
x1_train, x1_val = train_test_split(x1_train, test_size=0.2, random_state=66, shuffle=True)

x_predict = np.array([[89800,91800,88000,88000,-1.9,-4949552,-3522801]])
x1_predict = np.array([[4420,4545,4405,4515,0,1309200,-457634,0]])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_predict = scaler.transform(x_predict)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)
x_predict = x_predict.reshape(1,x_train.shape[1],1)

scaler.fit(x1_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
x1_val = scaler.transform(x1_val)
x1_predict = scaler.transform(x1_predict)

x1_train = x1_train.reshape(x1_train.shape[0],x1_train.shape[1],1)
x1_test = x1_test.reshape(x1_test.shape[0],x1_test.shape[1],1)
x1_val = x1_val.reshape(x1_val.shape[0],x1_val.shape[1],1)
x1_predict = x1_predict.reshape(1,x1_train.shape[1],1)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout

# 1
input1 = Input(shape = (7,1))
dense1=LSTM(500, activation='relu')(input1)
dense1=Dropout(0.2)(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dropout(0.2)(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dropout(0.2)(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dropout(0.2)(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dropout(0.2)(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dropout(0.2)(dense1)
dense1=Dense(500, activation='relu')(dense1)
dense1=Dense(1)(dense1)


# 2
input2 = Input(shape = (8,1))
dense2=LSTM(500, activation='relu')(input1)
dense2=Dropout(0.2)(dense2)
dense2=Dense(500, activation='relu')(dense2)
dense2=Dropout(0.2)(dense2)
dense2=Dense(500, activation='relu')(dense2)
dense2=Dropout(0.2)(dense2)
dense2=Dense(500, activation='relu')(dense2)
dense2=Dropout(0.2)(dense2)
dense2=Dense(500, activation='relu')(dense2)
dense2=Dropout(0.2)(dense2)
dense2=Dense(500, activation='relu')(dense2)
dense2=Dropout(0.2)(dense2)
dense2=Dense(500, activation='relu')(dense2)
dense2=Dense(1)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
merge1=Dense(100)(merge1)
merge1=Dropout(0.2)(merge1)
merge1=Dense(100)(merge1)
merge1=Dense(100)(merge1)
output1=Dense(2)(merge1)

model = Model(inputs=[input1,input2], outputs=output1)

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)
es = EarlyStopping(monitor='val_loss', patience=200, mode='auto')
model.fit([x_train, x1_train], y_train, batch_size=64, epochs=30000, validation_data=([x_val,x1_val], y_val), callbacks=[es, reduce_lr])
model.save('./samsung/save/samsung_fitsave/sam_210120_fitsave1.h5')

result = model.evaluate([x_test,x1_test], y_test, batch_size=64)
print('loss : ', result)
y_predict = model.predict([x_predict, x1_predict])
print(y_predict)

# loss :  [1105628.375, 796.3155517578125]
# [[87871.875 88384.62 ]]'''