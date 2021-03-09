# 한개는 Dense 한개는 LSTM 앙상블 모델 구성
import numpy as np
from numpy import array
# 1. 데이터
x1 = array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[20,30],[30,40],[40,50]])
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y1 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]])
y2 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x1_predict = array([55,65])
x2_predict = array([65,75,85])

print(x1.shape) #(13,2)
print(x2.shape) #(13,3)
print(y1.shape)  #(13,3)
print(y2.shape)  #(13, )
x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
x2 = x2.reshape(x2.shape[0],x2.shape[1],1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, shuffle=False, train_size=0.8, random_state=33)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, shuffle=False, train_size=0.8, random_state=33)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

# 1
input1 = Input(shape=(2,1))
dense1 = LSTM(10, activation='relu')(input1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(5, activation='relu')(dense1)
#output1 = Dense(3)(dense1)

# 2
input2 = Input(shape=(3,1))
dense2 = LSTM(10, activation='relu')(input2)
dense2 = Dense(10, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
#output2 = Dense(3)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(10)(merge1)
middle1 = Dense(50)(middle1)
# 1
output1 = Dense(30)(middle1)
output1 = Dense(3)(output1)
# 2
output2 = Dense(10)(middle1)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(1)(output2)

# 모델 선언
model = Model(inputs = [input1, input2], outputs = [output1, output2])


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=500, batch_size=13, validation_split=0.2, verbose=1)

#4. 예측, 평가
loss = model.evaluate([x1_test, x2_test],[y1_test, y2_test], batch_size=1)

print("model.metrics_name : ", model.metrics_names)
print(loss)

x1_predict=x1_predict.reshape(1,2,1)
x2_predict=x2_predict.reshape(1,3,1)

y_predict= model.predict([x1_predict, x2_predict])
print(y_predict)

# model.metrics_name :  ['loss', 'dense_7_loss', 'dense_11_loss', 'dense_7_mae', 'dense_11_mae']
# [3704.917236328125, 70.4660415649414, 3634.451171875, 8.070475578308105, 59.720550537109375]
# [array([[76.18571 , 85.05692 , 92.485115]], dtype=float32), array([[8.893562]], dtype=float32)]