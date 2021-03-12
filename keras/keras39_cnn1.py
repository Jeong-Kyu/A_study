from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,Flatten, MaxPool2D

model  = Sequential()
model.add(Conv2D(filters = 10,kernel_size=(2,2), strides=1, padding='same', input_shape = (7,7,1)))  # (input_dim * kernel_size + bias)*filter
#strides = 얼마나 건너서 자를건지 2 / (2,3)
model.add(MaxPool2D(pool_size=(2,3)))  # 2 / 3 / (2,3)
model.add(Conv2D(9,(2,2), padding='valid'))
# model.add(Conv2D(9,(2,3)))
# model.add(Conv2D(8,2))
model.add(Flatten())
model.add(Dense(1))

model.summary()