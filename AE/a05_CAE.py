# 4번 카피
# CNN으로 딥하게 구성

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(60000,28,28).astype('float32')/255
x_test = x_test.reshape(10000,28,28)/255

# print(x_train[0])
# print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten,Conv2D,BatchNormalization,LeakyReLU,Conv2DTranspose,Dropout

# def autoencoder(hidden_layer_size):
#     model = Sequential()
#     model.add(Conv1D(filters=hidden_layer_size, kernel_size=2 ,activation='relu', input_shape=(28,28)))
#     model.add(Flatten())
#     model.add(Dense(64))
#     model.add(Dense(32))
#     model.add(Dense(64))
#     model.add(Dense(128))
#     model.add(Dense(64))
#     model.add(Dense(32))
#     model.add(Dense(units=784, activation='sigmoid'))
#     return model

# model = autoencoder(hidden_layer_size=128)
# model.summary()

def autoencoder():
    inputs = Input(shape=(28,28,1))
    x = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x_1 = x
    x = Conv2D(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x_2 = x
    x = Conv2DTranspose(filters=64,kernel_size=4,strides=2,use_bias=False,padding='same')(x+x_2)
    x = Dropout(0.4)(x)
    x = LeakyReLU()(x)
    x = x
    x = Conv2DTranspose(filters=1,kernel_size=4,strides=2,use_bias=False,padding='same')(x+x_1)
    x = Dropout(0.4)(x)
    x = LeakyReLU()(x)
    x = x
    outputs = x
    model = Model(inputs = inputs,outputs=outputs)


    return model

model = autoencoder()#hidden_layer_size=128)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.compile(optimizer='adam', loss='mse', metrics=['acc'])

model.fit(x_train, x_train, epochs=10)
output = model.predict(x_test)

import matplotlib.pyplot as plt
import random
fig, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10)) = \
    plt.subplots(2,5,figsize=(20,7))

random_images =random.sample(range(output.shape[0]), 5)

for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap = 'gray')
    if i == 0 :
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap = 'gray')
    if i == 0 :
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()