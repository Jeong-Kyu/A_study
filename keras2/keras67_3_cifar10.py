# cifar10 flow
# ImageDataGenerator fit generator

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D,Flatten, BatchNormalization

from tensorflow.keras.datasets import cifar10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(128,(3,3), input_shape=(32,32,3)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3)))
model.add(Dropout(0.2))
model.add(Conv2D(16,(3,3)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(10, activation='softmax'))
epochs = 10
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# fits the model on batches with real-time data augmentation:
model.fit(datagen.flow(x_train, y_train, batch_size=32),
          steps_per_epoch=len(x_train) / 32, epochs=epochs)
# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

# 1/1 [==============================] - 0s 988us/step - loss: 2.0558 - acc: 0.2500