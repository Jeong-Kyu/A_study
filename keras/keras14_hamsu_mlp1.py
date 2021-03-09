#다:1 mlp
# keras10_mlp2.py 를 함수형으로

import numpy as np

#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array(range(711, 811))

x = np.transpose(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size=0.2, shuffle=True, random_state=66)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
#from keras.layers import Dense  --> 가능하지만 느리다.

input1 = Input(shape = (3,))
a1 = Dense(3, activation='relu')(input1)
a2 = Dense(5)(a1)
a3 = Dense(5)(a2)
output1 = Dense(1)(a3)
model = Model(inputs = input1, outputs = output1)
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x, y)
print('loss = ', loss)
print('mae = ', mae)

y_predict= model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)