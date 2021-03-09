#1:다 mlp


#실습 : 코드를 완성할 것
#mlp4 pred값을 도출할 것

import numpy as np

#1. 데이터
x = np.array([range(100)])
y = np.array([range(711, 811), range(1,101), range(100)])

x_pred = np.array([(101)])

print(x. shape) #--> (1,100)
print(y. shape) #--> (3,100)

x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size=0.2, shuffle=True, random_state=66)


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import Dense  --> 가능하지만 느리다.

model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(3))

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

y_pred= model.predict(x_pred)
print(y_pred)