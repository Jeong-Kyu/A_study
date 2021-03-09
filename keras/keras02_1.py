#네이밍 룰 (오류는 아니지만 약속)
#keras00_numpy.py(python,C)    keras00Numpy.py(java)

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential

#from tensorflow.keras import models
#model = models.Sequential()

#from tensorflow import keras
#model = keras.models.Sequential()

from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

x_test = np.array([6,7,8])
y_test = np.array([6,7,8])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(10000))
model.add(Dense(10000))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가,예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
result = model.predict([9])
print('result : ',result)
