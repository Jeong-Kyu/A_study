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
# 신경망을 구성한다 input-> 5-> 3-> 4-> 1(output)

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, SGD 
optimizer=Adam(learning_rate=0.1)
# optimizer=SGD(learning_rate=0.1)
model.compile(loss = 'mse', optimizer=optimizer)
model.fit(x, y, epochs=1000, batch_size=1)
# MSE(평균제곱오차)
# NUMPY(고성능 수치계산 - 행렬)

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
print('loss : ', loss)

x_pred = np.array([4])
result = model.predict(x_pred)
#result = model.predict([x])
print('result : ', result)


#데이터 -> 모델구성 -> 컴파일,훈련 -> 평가,예측