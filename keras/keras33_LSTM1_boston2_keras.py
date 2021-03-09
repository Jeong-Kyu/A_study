# 텐서플로 데이터셋
# LSRM
# DENSE
# 회기모델

import numpy as np
from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# print(x_train.shape)
# print(x_test.shape)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
# print(x_train.shape)
# print(x_test.shape)
# print(x_val.shape)

x_train = x_train.reshape(323,13,1)
x_test = x_test.reshape(102,13,1)
x_val = x_val.reshape(81,13,1)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM
inputs = Input(shape=(13,1))
dense1 = LSTM(50, activation='relu')(inputs)
dense1 = Dense(100)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dense(200)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dense(50)(dense1)
outputs = Dense(1)(dense1)

model = Model(inputs=inputs, outputs=outputs)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=16, epochs=300, validation_data=(x_val, y_val))

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=16)
y_predict = model.predict(x_test)
print("loss, mae : ", loss, mae)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
#print("mse : ", mean_squared_error(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 이걸로 만들어라 !!! 아까처럼 사이킷런으로 당기지 않는다


# early stopping 전
# loss, mae :  11.031598091125488 2.355877637863159
# RMSE :  3.3213850693202898
# R2 :  0.8674785107264735'''

# LSTM
# loss, mae :  25.439069747924805 3.320350170135498
# RMSE :  5.043716229669135
# R2 :  0.694402963609053