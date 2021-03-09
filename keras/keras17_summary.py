import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1, activation='linear'))
model.add(Dense(3, activation='linear'))
model.add(Dense(4))
model.add(Dense(1))


model.summary()

# + 바이어스

#실습2 + 과제
# ensemble 1,2,3,4에 대해 서머리를 계산하고
# 이해한 것을 과제로 제출할것
# layer를 만들 때 name대해 확인하고 설명