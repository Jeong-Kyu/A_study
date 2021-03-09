#보스턴 집값
#실습:모델구성
import numpy as np
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape) #(506, 13)
print(y.shape) #(506,   )

#print("===================")
#print(x[:5]) # 0~4
#print(y[:10]) 
#print(np.max(x), np.min(x)) # max값 min값
#print(dataset.feature_names)
#print(dataset.DESCR) #묘사

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state= 66) #random_state 랜덤변수 고정 

#2. 모델링
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(45, activation='relu')(dense1) 
dense3 = Dense(40, activation='relu')(dense2)
dense4 = Dense(35, activation='relu')(dense3)
dense5 = Dense(30, activation='relu')(dense4)
dense6 = Dense(25, activation='relu')(dense5)
dense7 = Dense(25, activation='relu')(dense6)
dense8 = Dense(25, activation='relu')(dense7)
dense9 = Dense(25, activation='relu')(dense8)
outputs = Dense(1)(dense9)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=7, validation_split=0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test) 
#print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))  #sqrt는 루트
print("RMSE :" , RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2 )

# loss :  16.55705451965332
# mae :  3.3871774673461914
# RMSE : 4.069036165639308
# R2 :  0.8019086688524137







#전처리가 된 데이터(정규화)
#[6.3200e-03 1.8000e+01 2.3100e+00 0.0000e+00 5.3800e-01 6.5750e+00] = 되어있지않음...

'''
    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
'''