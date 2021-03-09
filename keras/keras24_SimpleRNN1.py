#simpleRNN

# 1. 데이터
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])

print(x.shape) #(4,3)
print(y.shape) #(4, )

x = x.reshape(4, 3, 1)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
model = Sequential()
model.add(SimpleRNN(10, activation='relu', input_shape = (3,1)))   #LSTM - 3차원을 받음
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

model.summary()
# 4 * (n * m + 1) * m
# n: 인풋 벡터의 차원
# m: 레이어의 유닛갯수




# 3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가 예측
loss = model.evaluate(x, y)
print(loss)

x_pred = np.array([5,6,7]) #(3, )   ->  (1 ,3, 1)
x_pred = x_pred.reshape(1, 3, 1)
result = model.predict(x_pred)
print(result)

#LSTM
# 0.015908680856227875
# [[7.287629]]

# #SimpleRNN
# 0.0606086328625679
# [[8.387222]]