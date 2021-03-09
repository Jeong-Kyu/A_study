#실습

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(60000,28,28).astype('float32')/255
x_test = x_test.reshape(10000,28,28)/255

x_train_noised = x_train + np.random.normal(0,0.1,size = x_train.shape)
x_test_noised = x_test + np.random.normal(0,0.1,size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout

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

model = autoencoder() # 95% pca수치
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])
model.fit(x_train_noised,x_train,epochs=10)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3,5,figsize=(20,7))

random_images = random.sample(range(output.shape[0]),5)

# 원본
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()