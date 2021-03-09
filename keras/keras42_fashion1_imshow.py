import numpy as np
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)


import matplotlib.pyplot as plt


(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(x_train[0])
print('y_train[0] : ', y_train[0])
print(x_train[0].shape)

plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])

plt.show()