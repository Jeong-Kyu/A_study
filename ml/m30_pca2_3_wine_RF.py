import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
datasets = load_wine()
x = datasets.data
y = datasets.target

# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print('cumsum : ', cumsum)

# d = np.argmax(cumsum>0.99)+1
# print('cumsum >= 0.99', cumsum>=0.99)
# print('d : ', d)

# import matplotlib.pylab as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

pca =PCA(n_components = 9)
x2 = pca.fit_transform(x)

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

# [0.12823997 0.03401374 0.06991065 0.17157269 0.14945659 0.09712037
#  0.13586567 0.21382033]
# acc :  0.9444444444444444
# RF
# [0.13354783 0.03389515 0.01299818 0.02826238 0.02687611 0.04924811
#  0.17041261 0.00576612 0.03144947 0.13331536 0.10217762 0.09364459
#  0.17840645]
# acc :  0.9722222222222222
# XGB
# [0.06830431 0.04395564 0.00895185 0.         0.01537293 0.00633501
#  0.07699447 0.00459099 0.00464443 0.08973485 0.01806366 0.5588503
#  0.10420163]
# acc :  0.9444444444444444