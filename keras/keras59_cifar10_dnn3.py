# 다차원 댄스 모델
# 32,32,3 -> ,32,32,3

#45번 카피

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape)#(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)#(10000, 28, 28) (10000,)

# print(x_train[0])
# print('y_train[0] : ', y_train[0])
# print(x_train[0].shape)#(28, 28)

# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()
y_test1=y_test
# x_train = x_train.reshape(60000,28,28,1).astype('Float32')/255.
x_train = x_train.reshape(50000,32,32,3)/255.
x_test = x_test.reshape(10000,32,32,3)/255.
# x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

y_train = x_train
y_test = x_test

print(y_train.shape) # (60000,28,28,1)
print(y_test.shape) # (10000,28,28,1)


# #OnehotEncoding
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# encoder = OneHotEncoder()
# y_train = encoder.fit_transform(y_train.reshape(-1,1)).toarray()
# y_test = encoder.fit_transform(y_test.reshape(-1,1)).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

model = Sequential()
# model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
# model.add(MaxPool2D(pool_size=2))
model.add(Dense(64, input_shape=(32,32,3)))
model.add(Dropout(0.5))
# model.add(Conv2D(1,(2,2)))
# model.add(Conv2D(1,(2,2)))
model.add(Dense(64))
# model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(1))
model.summary()

##완성하기
#지표는 acc 0.985
#x_test 10개 y_pred 10개 출력
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = '../data/modelcheckpoint/k57_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor='val loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=16, epochs=1, validation_split=0.5, callbacks=[es, cp, reduce_lr])

loss, acc = model.evaluate(x_test, y_test, batch_size=200)
print('loss : ', loss)
print('accuracy : ', acc)

y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred.shape)


'''
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic"
plt.figure(figsize=(10, 6))

plt.subplot(2,1,1)  # 2행 1열 중 첫번째
plt.plot(hist.history['loss'], marker= '.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker= '.', c='blue', label='val_loss')
plt.grid()

plt.title('손실Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') #upper right가 없을 경우 빈 공간을 찾아감

plt.subplot(2,1,2) # 2행 2열 중 두번쨰
plt.plot(hist.history['accuracy'], marker= '.', c='red')
plt.plot(hist.history['val_accuracy'], marker= '.', c='blue')
plt.grid()

plt.title('정확Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy','val_accuracy']) #legend에 직접 라벨명을 넣으면 표기

plt.show()'''