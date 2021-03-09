import numpy as np
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape)# (50000,32,32,3)
# print(x_test.shape) # (10000,32,32,3)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=66, shuffle=True)

x_train = x_train.reshape(40000,32*32,3)/225.
x_test = x_test.reshape(10000,32*32,3)/225.
x_val = x_val.reshape(10000,32*32,3)/225.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D,Flatten

#2. 모델링
model  = Sequential()
model.add(Conv1D(filters = 10,kernel_size=(5), input_shape = (32*32,3)))
model.add(MaxPool1D(pool_size=(3)))
model.add(Flatten())
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics = ['mae'])
'''
EarlyStopping
'''
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto') #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
model.fit(x_test, y_test, epochs=10, batch_size=1, validation_data= (x_val, y_val), callbacks = [early_stopping])

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

# cnn
# loss :  6.173558235168457
# mae :  2.0689027309417725
# RMSE : 2.4846644
# R2 :  0.2516900640863633

# conv1
# loss :  5.982679843902588
# mae :  2.0267210006713867
# RMSE : 2.4459517
# R2 :  0.2748266900430335