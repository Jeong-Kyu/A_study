import numpy as np
from tensorflow.keras.datasets import cifar100
(x_trian, y_train), (x_test, y_test) = cifar100.load_data()

print(x_trian.shape)# (50000,32,32,3)
print(x_test.shape) # (10000,32,32,3)



