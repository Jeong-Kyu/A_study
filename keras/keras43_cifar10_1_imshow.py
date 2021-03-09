import numpy as np
from tensorflow.keras.datasets import cifar10
(x_trian, y_train), (x_test, y_test) = cifar10.load_data()

print(x_trian.shape)# (50000,32,32,3)
print(x_test.shape) # (50000,1)



