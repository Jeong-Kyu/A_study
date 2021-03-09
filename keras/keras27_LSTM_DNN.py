# keras23_LStM3_scale DNN으로
#결과비교

#DNN으로 23번 파일보다 loss를 좋게 만들것

import numpy as np
# 1datetime A combination of a date and a time. Attributes: ()
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

print(x.shape) #(13,3)
print(y.shape) #(13, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

x_pred=x_pred.reshape(1,3)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) #x_train만 trans 후 바뀐수치 x_train에 다시넣기
x_test = scaler.transform(x_test) #x_test 따로 trans  후 바뀐수치 x_test에 다시넣기
x_pred = scaler.transform(x_pred)



x = x.reshape(13,3)
y = y.reshape(13,1)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(3,)))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(50))
model.add(Dense(1000))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, epochs=500, batch_size=1)

loss=model.evaluate(x_test,y_test)
print(loss)


y_pred=model.predict(x_pred)
print(y_pred)

# RNN
# 0.07824796438217163
# [[80.906136]]

# 튜닝전 DNN
# 0.11495145410299301
# [[79.29692]]

# 튜닝후 MinMax 추가
# 0.05811046063899994
# [[79.51945]]
