#다:다 mlp


#실습
#x는 (100, 5) 데이터 구성
#y는 (100, 2) 데이터 구성




import numpy as np

#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101), range(111, 211), range(11, 111)])
y = np.array([range(711, 811), range(1,101)])

print(x. shape) #--> (3,100)
print(y. shape) #--> (3,100)

x = np.transpose(x)
y = np.transpose(y)

print(x. shape)
print(x)        #--> (100,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size=0.2, shuffle=True, random_state=66)


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.layers import Dense  --> 가능하지만 느리다.

model = Sequential()
model.add(Dense(10, input_dim=5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
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

x_final = np.array([99, 400, 100, 210, 110])
x_final = x_final.reshape(1,5)
print(x_final.shape)
print(x_final)

y_final= model.predict(x_final)
print(y_final) 

#loss =  1.0025023033932712e-09
#mae =  2.122223304468207e-05
#RMSE :  3.16623160710702e-05
#mse :  1.0025022589843502e-09
#R2 :  0.9999999999987316
#(1, 5)
#[[ 99 400 100 210 110]]
#[[809.99994  99.99999]]
