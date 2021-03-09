import numpy as np
from sklearn.datasets import load_linnerud

dataset=load_linnerud()
x = dataset.data
y = dataset.target

# print(x.shape) #(20, 3)
# print(y.shape) #(20, 3)
# print(dataset.DESCR) # 다 : 다

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=66)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1=Input(shape=(3,))
dense1=Dense(5, activation='relu')(input1)
dense1=Dense(10, activation='relu')(input1)
dense1=Dense(5, activation='relu')(input1)
dense1=Dense(30, activation='relu')(input1)
dense1=Dense(10, activation='relu')(input1)
dense1=Dense(50, activation='relu')(dense1)
output1=Dense(3, activation='relu')(dense1)

model = Model(inputs=input1, outputs=output1)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='auto')

hist = model.fit(x_train, y_train, batch_size=10, epochs=1000, validation_data=(x_val, y_val), callbacks=[es])

loss = model.evaluate(x_test, y_test, batch_size=10)
print("loss = ", loss)
y_pred = model.predict(x_test)
print(y_pred)

import matplotlib.pyplot as plt

# plt.plot(x, y)
# plt.show()
# plt.plot(y값)--흐름에 따른 변화량 보여줌
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])


plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss']) # 순서대로 설명
plt.show()

