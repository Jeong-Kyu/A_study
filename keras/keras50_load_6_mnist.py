import numpy as np

x_train = np.load('../data/npy/mnist_x_train.npy')
x_test = np.load('../data/npy/mnist_x_test.npy')
y_train = np.load('../data/npy/mnist_y_train.npy')
y_test = np.load('../data/npy/mnist_y_test.npy')


x_train = x_train.reshape(60000,28,28,1)/255.
x_test = x_test.reshape(10000,28,28,1)/255.
# x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

#OnehotEncoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = encoder.fit_transform(y_test.reshape(-1,1)).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=2))
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor='val loss', patience=5)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=128, epochs=30, validation_split=0.2, callbacks=[es, cp])

loss, acc = model.evaluate(x_test, y_test, batch_size=200)
print('loss : ', loss)
print('accuracy : ', acc)

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
plt.legend(loc='upper right')

plt.subplot(2,1,2) # 2행 2열 중 두번쨰
plt.plot(hist.history['accuracy'], marker= '.', c='red', label='accuracy')
plt.plot(hist.history['val_accuracy'], marker= '.', c='blue', label='val_accuracy')
plt.grid()

plt.title('정확Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()