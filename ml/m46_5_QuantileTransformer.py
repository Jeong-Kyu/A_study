# 18-3 copy

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

#x = x /711
#x = (x - 최소) / (최대 - 최소)
#  = (x - np.min(x)) / (np.max(x) - np.min(x))
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = QuantileTransformer() # 분위수 1000개 / 디폴트 : 균등분포
# scaler = QuantileTransformer(output_distribution='normal') # 정규분포
scaler.fit(x)
x = scaler.transform(x)
print(np.max(x), np.min(x)) # 711.0.0.0 -> 1.0.0.0


'''
X를 전체로 전처리해버림
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state= 66) #random_state 랜덤변수 고정 



#2. 모델링
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(45, activation='relu')(dense1) 
dense3 = Dense(40, activation='relu')(dense2)
dense4 = Dense(35, activation='relu')(dense3)
dense5 = Dense(30, activation='relu')(dense4)
dense6 = Dense(30, activation='relu')(dense5)
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

#전처리 전
# loss :  16.55705451965332
# mae :  3.3871774673461914
# RMSE : 4.069036165639308
# R2 :  0.8019086688524137

#전처리 후
# loss :  11.465134620666504
# mae :  2.5706095695495605
# RMSE : 3.386020620416784
# R2 :  0.8628292327610475

# standard
# loss :  13.358779907226562
# mae :  2.208256483078003
# RMSE : 3.654966366429312
# R2 :  0.8401733707120181

# robust
# loss :  9.583504676818848
# mae :  2.2211551666259766
# RMSE : 3.0957235781194417
# R2 :  0.8853413773425858

# quantiletransformer
# loss :  9.396515846252441
# mae :  2.197286367416382
# RMSE : 3.0653735409479426
# R2 :  0.8875785507723567