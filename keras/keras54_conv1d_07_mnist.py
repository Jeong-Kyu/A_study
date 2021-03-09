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
x_train = x_train.reshape(60000,28,28)/255.
x_test = x_test.reshape(10000,28,28)/255.
# x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

#OnehotEncoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = encoder.fit_transform(y_test.reshape(-1,1)).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv1D(filters=100, kernel_size=2,  input_shape=(28,28)))
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.2))
# model.add(Conv2D(???????))
model.add(Flatten())
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(10, activation='softmax'))

##완성하기
#지표는 acc 0.985
#x_test 10개 y_pred 10개 출력
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val loss', patience=5)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.2, callbacks=[es])

loss, acc = model.evaluate(x_test, y_test, batch_size=200)
print('loss : ', loss)
print('accuracy : ', acc)

# loss :  0.17273838818073273
# accuracy :  0.9487000107765198