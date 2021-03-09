# 주말과제
# LSTM 모델로 구성(764,1)
# LSTM 모델로 구성(28*14,2)
# LSTM 모델로 구성(28*7,4)....

# 주말과제
# dense 모델로 구성 input_shape(28*28,)

# 인공지능계의 helli world - mnist

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)#(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)#(10000, 28, 28) (10000,)

print(x_train[0])
print('y_train[0] : ', y_train[0])
print(x_train[0].shape)#(28, 28)

# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()
y_test1=y_test
# x_train = x_train.reshape(60000,28,28,1).astype('Float32')/255.
x_train = x_train.reshape(60000,28*28,1)/255.
x_test = x_test.reshape(10000,28*28,1)/255.
# x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

#OnehotEncoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = encoder.fit_transform(y_test.reshape(-1,1)).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, LSTM

model = Sequential()
model.add(LSTM(30,input_shape=(28*28,1), activation='relu'))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(10, activation='softmax'))

##완성하기
#지표는 acc 0.985
#x_test 10개 y_pred 10개 출력

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=128, epochs=100)

loss, acc = model.evaluate(x_test, y_test, batch_size=200)
print('loss : ', loss)
print('acc : ', acc)
y_pred = model.predict(x_test[:10])
print(y_test1[:10])
print(np.argmax(y_pred,axis=-1))

# cnn
# loss :  0.1353285312652588
# acc :  0.9728000164031982
# [7 2 1 0 4 1 4 9 5 9]
# [7 2 1 0 4 1 4 9 6 9]

# dnn
# loss :  0.336788147687912
# acc :  0.9638000130653381
# [7 2 1 0 4 1 4 9 5 9]
# [7 2 1 0 4 1 4 9 5 9]