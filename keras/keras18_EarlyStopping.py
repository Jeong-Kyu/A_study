#보스턴 집값
#실습:EarlyStopping
# 과적합 구간
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

#1_2. 데이터 전처리(MinMaxScalar)
#ex 0~711 = 최댓값으로 나눈다  0~711/711
# X - 최소값 / 최대값 - 최소값
print("===================")
print(x[:5]) # 0~4
print(y[:10]) 
print(np.max(x), np.min(x)) # max값 min값
print(dataset.feature_names)
#print(dataset.DESCR) #묘사

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state= 66) #random_state 랜덤변수 고정 
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=66, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_tranin = scaler.transform(x_train) #x_train만 trans 후 바뀐수치 x_train에 다시넣기
x_test = scaler.transform(x_test) #x_test 따로 trans  후 바뀐수치 x_test에 다시넣기

#2. 모델링
input1 = Input(shape=(13,))
dense1 = Dense(36, activation='relu')(input1)
dense2 = Dense(40, activation='relu')(dense1) 
dense3 = Dense(40, activation='relu')(dense2)
dense4 = Dense(40, activation='relu')(dense3)
outputs = Dense(1)(dense4)
model = Model(inputs = input1, outputs = outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics = ['mae'])
'''
EarlyStopping
'''
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto') #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
model.fit(x_test, y_test, epochs=2000, batch_size=1, validation_data= (x_val, y_val), callbacks = [early_stopping])

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

#Early_stopping적용 patience=10
# loss :  2.5246715545654297
# mae :  1.1248575448989868
# RMSE : 1.5889214892936647
# R2 :  0.9619927199166804

#Early_stopping적용 patience=20
# loss :  1.4020894765853882
# mae :  0.9438440203666687
# RMSE : 1.1840988404777681
# R2 :  0.978892450054018