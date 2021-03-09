# 사이킷런 데이터셋
# LSRM
# DENSE
# 회기모델

import numpy as np
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
# print(x.shape) #(506, 13)
# print(y.shape) #(506,   )




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state= 66) #random_state 랜덤변수 고정 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=66, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) #x_train만 trans 후 바뀐수치 x_train에 다시넣기
x_test = scaler.transform(x_test) #x_test 따로 trans  후 바뀐수치 x_test에 다시넣기
x_val = scaler.transform(x_val) #x_test 따로 trans  후 바뀐수치 x_test에 다시넣기


x_train = x_train.reshape(323,13,1)
x_test = x_test.reshape(102,13,1)
x_val = x_val.reshape(81,13,1)


#2. 모델링
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM 
input1 = Input(shape=(13,1))
dense1 = LSTM(36, activation='relu')(input1)
dense2 = Dense(40, activation='relu')(dense1) 
dense3 = Dense(100, activation='relu')(dense2)
dense4 = Dense(100, activation='relu')(dense3)
dense4 = Dense(100, activation='relu')(dense4)
dense4 = Dense(100, activation='relu')(dense4)
dense4 = Dense(100, activation='relu')(dense4)
dense4 = Dense(40, activation='relu')(dense4)
outputs = Dense(1)(dense4)
model = Model(inputs = input1, outputs = outputs)


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics = ['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, epochs=500, batch_size=10, validation_data= (x_val, y_val))



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

#전처리 전
# loss :  16.55705451965332
# mae :  3.3871774673461914
# RMSE : 4.069036165639308
# R2 :  0.8019086688524137

#통째로 전처리
# loss :  11.465134620666504
# mae :  2.5706095695495605
# RMSE : 3.386020620416784
# R2 :  0.8628292327610475

#제대로 전처리(?)
# loss :  531.5300903320312
# mae :  21.24960708618164
# RMSE : 23.054936080104717
# R2 :  -5.359313211830821

#발리데이션 test분리
# loss :  5.44482421875
# mae :  1.7919334173202515
# RMSE : 2.3334145056348183
# R2 :  0.9430991642272919

# # LSTM 적용
# loss :  14.602876663208008
# mae :  2.8887219429016113
# RMSE : 3.8213711087881648
# R2 :  0.8252887776232067

# LSTM 튠적용
# loss :  10.266945838928223
# mae :  2.4027605056762695
# RMSE : 3.2042074819775657
# R2 :  0.8771645755065441