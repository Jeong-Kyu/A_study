import numpy as np
# data(3,6,5,4,2)
# 1. sk onehot
# 2. tf kes to cate
# 1)결과치 2)shape
x_train=np.array([3,6,5,4,2])
y_train=np.array([3,6,5,4,2])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
x_train = encoder.fit_transform(x_train.reshape(-1,1)).toarray()
print(x_train)
print(x_train.shape)

# [[0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 1.]
#  [0. 0. 0. 1. 0.]
#  [0. 0. 1. 0. 0.]
#  [1. 0. 0. 0. 0.]]
# (5, 5)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train,7)
print(y_train)
print(y_train.shape)

# [[0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0.]]
# (5, 7)