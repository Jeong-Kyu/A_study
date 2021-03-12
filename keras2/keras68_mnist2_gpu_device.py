import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devixe('GPU')
if gpus :
    try:
        tf.config.experimental.set_visible_devices(gpus[1],'GPU')
    except RuntimeError as e :
        print(e)

(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)#(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)#(10000, 28, 28) (10000,)

print(x_train[0])
print('y_train[0] : ', y_train[0])
print(x_train[0].shape)#(28, 28)

y_test1=y_test
x_train = x_train.reshape(60000,28,28,1)/255.
x_test = x_test.reshape(10000,28,28,1)/255.


#OnehotEncoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = encoder.fit_transform(y_test.reshape(-1,1)).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.regularizers import l1, l2, l1_l2
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32,(2,2), kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu')) # relu 계열 - he_normal  sigmoid,softmax계열 - xavier

model.add(Conv2D(32,(2,2), kernel_regularizer=l1(l1=0.01)))
model.add(Dropout(0.2))

model.add(Conv2D(32,(2,2), strides=2))
model.add(MaxPool2D(pool_size=2))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val loss', patience=5)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.2, callbacks=[es])

loss, acc = model.evaluate(x_test, y_test, batch_size=200)
print('loss : ', loss)
print('accuracy : ', acc)

# loss :  0.09585337340831757
# accuracy :  0.9824000000953674