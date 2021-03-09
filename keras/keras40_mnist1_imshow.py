# 인공지능계의 helli world - mnist

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print(x_train[0])
print('y_train[0] : ', y_train[0])
print(x_train[0].shape)

plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])

plt.show()