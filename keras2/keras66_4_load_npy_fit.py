import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy')
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dropout,Dense,Flatten
model = Sequential()
model.add(Conv2D(256,(3,3), input_shape=(150,150,3)))
model.add(Dropout(0.2))
model.add(Conv2D(128,(3,3)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(16, activation = 'sigmoid'))

model.add(Dense(1, activation = 'sigmoid'))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='loss', patience=10, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test), callbacks=[es,reduce_lr])


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss & acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc']) # 순서대로 설명
plt.show()