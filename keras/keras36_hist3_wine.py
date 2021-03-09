import numpy as np

from sklearn.datasets import load_wine

dataset = load_wine()


x = dataset.data
y = dataset.target

# print(x)
# print(y)
# print(x.shape) #(178, 13)
# print(y.shape) #(178, )

# 실습 DNN 완성

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66 )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x = scaler.transform(x)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 2차원 데이터로 변환하기
y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
y_val = y_val.reshape(-1,1)

# print(y_test)
# print(y_train)
# print(y_val)


# 원-핫 인코딩 적용
encoder = OneHotEncoder()
encoder.fit(y_test)
y_test = encoder.transform(y_test)
encoder.fit(y_train)
y_train = encoder.transform(y_train)
encoder.fit(y_val)
y_val = encoder.transform(y_val)

y_test=y_test.toarray()
y_train=y_train.toarray()
y_val=y_val.toarray()

# 2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(10, input_shape=(13,), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

hist = model.fit(x_train, y_train, epochs=500, batch_size = 10, validation_data=(x_val, y_val), callbacks = [early_stopping])

print(hist)
print(hist.history.keys()) #loss, acc, val_loss, val_acc

print(hist.history['loss'])

# 그래프
import matplotlib.pyplot as plt

# plt.plot(x, y)
# plt.show()
# plt.plot(y값)--흐름에 따른 변화량 보여줌
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc']) # 순서대로 설명
plt.show()
