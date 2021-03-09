import numpy as np
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=66, shuffle=True)

x_train = x_train.reshape(48000,28,28,1)/225.
x_test = x_test.reshape(10000,28,28,1)/225.
x_val = x_val.reshape(12000,28,28,1)/225.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,Flatten

#2. 모델링
model  = Sequential()
model.add(Conv2D(filters = 10,kernel_size=(5,5), strides=1, padding='same', input_shape = (28,28,1)))  # (input_dim * kernel_size + bias)*filter
#strides = 얼마나 건너서 자를건지 2 / (2,3)
model.add(MaxPool2D(pool_size=(3,3)))  # 2 / 3 / (2,3)
# model.add(Conv2D(9,(2,3)))
# model.add(Conv2D(8,2))
model.add(Flatten())
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics = ['mae'])
'''
EarlyStopping
'''
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto') #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
model.fit(x_train, y_train, epochs=10, batch_size=1, validation_data= (x_val, y_val), callbacks = [early_stopping])

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
# loss :  1.6982073783874512
# mae :  0.9694334864616394
# RMSE : 1.3031527
# R2 :  0.7941567117430608