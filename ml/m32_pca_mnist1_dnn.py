import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
(x_train, y_train),(x_test, y_test) = mnist.load_data()


x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

x = x.reshape(70000,28*28)
# print(x.shape) (70000, 28, 28)

pca =PCA(n_components = 154)
x = pca.fit_transform(x)
# print(x.shape) (70000, 154)
