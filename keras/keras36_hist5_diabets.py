# hist를 이용한 그래프
# loss ,val_loss

import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#1. 데이터 
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])
print(x.shape, y.shape) #(442, 10) (442,)
print(np.max(x), np.min(x))
print(dataset.feature_names)
#print(dataset.DESCR)

#x = x / np.max(x)
#print(np.max(x), np.min(x)) # 정규화

#1_2. 데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size= 0.8, shuffle = True, random_state = 66 )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2. 모델링
model = Sequential()
model.add(Dense(100, input_dim=10, activation = 'relu')) # 기본값 : activation='linear' 
model.add(Dense(75, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
'''
EarlyStopping
'''
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=7, validation_data= (x_val, y_val), callbacks = [early_stopping])

print(hist)
print(hist.history.keys()) #loss, acc, val_loss, val_acc

print(hist.history['loss'])

# 그래프
import matplotlib.pyplot as plt

# plt.plot(x, y)
# plt.show()
# plt.plot(y값)--흐름에 따른 변화량 보여줌
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])


plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss']) # 순서대로 설명
plt.show()