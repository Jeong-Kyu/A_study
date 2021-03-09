# hist를 이용한 그래프
# loss ,val_loss, acc, val_acc

import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. 데이터
datesets = load_breast_cancer()
# print(datesets.DESCR)
# print(datesets.feature_names)

x = datesets.data
y = datesets.target
# print(x.shape) #(569, 30)
# print(y.shape) #(569,)
# print(x[:5])
# print(y)
# 전처리 알아서 MinMaxScaler, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,))) #히든없어도 가능
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))#히든없어도 가능
model.add(Dense(10, activation='relu'))#히든없어도 가능
model.add(Dense(1, activation='sigmoid')) #이진분류일 때 activation을 sigmoid로 사용
# linear -무한 ~ 무한 / relu 0 ~ 무한 / sigmoid 0 ~ 1

# 3. 컴파일, 훈련
                    # mean_squared_error                          #accuracy, mae
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# 이진분류일 때 loss를 binary_crossentropy로 사용  # metrics에 가급적 acc로 사용
# mse는 회귀모델
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

hist = model.fit(x_train, y_train, epochs=500, batch_size = 10, validation_split=0.2, callbacks = [early_stopping])


# 그래프
import matplotlib.pyplot as plt

# plt.plot(x, y)
# plt.show()
# plt.plot(y값)--흐름에 따른 변화량 보여줌
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc']) # 순서대로 설명
plt.show()