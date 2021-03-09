import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

datasets = load_diabetes()
x = datasets.data
y = datasets.target

pca =PCA(n_components = 7)
x2 = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)