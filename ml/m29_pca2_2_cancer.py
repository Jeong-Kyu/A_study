import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# pca =PCA(n_components = 7)
# x2 = pca.fit_trainsform(x)

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum : ', cumsum)

d = np.argmax(cumsum>0.99)+1
print('cumsum >= 0.99', cumsum>=0.99)
print('d : ', d)

import matplotlib.pylab as plt
plt.plot(cumsum)
plt.grid()
plt.show()