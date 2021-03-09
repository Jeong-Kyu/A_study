#validation_data만들기
#train_test_split사용


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(1,101))
y = np.array(range(101,201))

'''x_train = x[:60]   # 순서 0번째부터 59번째 까지 :: 값 1~60
x_val = x[60:80]
x_test = x[80:]
#리스트의 슬라이싱

y_train = y[:60]   # 순서 0번째부터 59번째 까지 :: 값 1~60
y_val = y[60:80]
y_test = y[80:]
#리스트의 슬라이싱'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

#4. 평가, 예측
loss, mae = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('mse : ', mae)

y_predict =model.predict(x_test)
print(y_predict)

#shuffle False
#loss :  0.0011673510307446122
#mse :  0.03406677395105362
#shuffle true
#loss :  0.05118560791015625
#mse :  0.20711974799633026
#validation = 0.2
#loss :  2.3858585357666016
#mse :  1.5445995330810547
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ", RMSE(y_test,y_predict))
print("mse : ", mean_squared_error(y_test,y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

#val
#loss :  7.36499547958374
#mse :  2.2938923835754395
#RMSE :  2.713852640551199
#mse :  7.364996154626715
#R2 :  0.9882664603729934