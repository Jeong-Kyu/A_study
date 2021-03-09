from sklearn.datasets import load_digits
import numpy as np

datasets = load_digits()
x = datasets.data
y = datasets.target

# print(x)
# print(y)
# print(x.shape)
# print(y.shape)
# print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=66)

# print(x_train.shape)
# print(x_test.shape)
# print(x_val.shape)

x_train = x_train.reshape(1149,8,8,1)/225.
x_test = x_test.reshape(360,8,8,1)/225.
x_val = x_val.reshape(288,8,8,1)/225.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(3,3), strides=1, padding='same', input_shape=(8,8,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
ES = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
MC = ModelCheckpoint('../data/test/test5_{epoch:02d}_{val_loss:.4f}.h5', monitor='val_loss', save_best_only=True, mode='auto')

model.compile(loss='mse',optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=30, validation_data=(x_val, y_val), epochs=100, callbacks=[ES,MC])

result = model.evaluate(x_test, y_test, batch_size=30)
print(result)



