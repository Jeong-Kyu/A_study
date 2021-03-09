import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
x = x.reshape(70000,28*28)
# print(x.shape) (70000, 28, 28)

# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print('cumsum : ', cumsum)

# d = np.argmax(cumsum>0.95)+1
# print('cumsum >= 0.95', cumsum>=0.95)
# print('d : ', d) # 154

# import matplotlib.pylab as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

pca =PCA(n_components = 154)
x = pca.fit_transform(x)
