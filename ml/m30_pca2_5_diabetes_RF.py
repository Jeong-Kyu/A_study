import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
datasets = load_diabetes()
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

pca =PCA(n_components = 3)
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

# [0.14478263 0.17663638 0.16863402 0.17178241 0.17188971 0.16627485]
# acc :  0.011235955056179775
# RF
# [0.11039481 0.01451288 0.12808419 0.09800651 0.10712409 0.11365845
#  0.12564916 0.04608623 0.12467218 0.13181149]
# acc :  0.011235955056179775