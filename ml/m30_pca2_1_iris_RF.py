import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd

datasets = load_iris()
x = datasets.data
y = datasets.target

# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print('cumsum : ', cumsum)

# d = np.argmax(cumsum>0.95)+1
# print('cumsum >= 0.95', cumsum>=0.95)
# print('d : ', d)

# import matplotlib.pylab as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

# cumsum :  [0.92461872 0.97768521 0.99478782 1.        ]
# cumsum >= 0.95 [False  True  True  True]
# d :  2
pca =PCA(n_components = 3)
x = pca.fit_transform(x)

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 44)
# 2.
model = XGBClassifier(n_jobs = -1, use_label_encoder=False)
# 3.
model.fit(x_train, y_train, eval_metric='logloss')
# 4.
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print('acc : ', acc)

# [0.32930836 0.195092   0.47559964]
# acc :  0.9333333333333333
# RF
# [0.59651981 0.21181421 0.19166599]
# acc :  0.9
# XGB
# [0.84933317 0.11487827 0.03578857]
# acc :  0.9333333333333333