import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
datasets = load_boston()
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
model = XGBRegressor(n_jobs = -1, use_label_encoder=False)
# 3.
model.fit(x_train, y_train, eval_metric='logloss')
# 4.
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print('acc : ', acc)

# [0.02633349 0.03025513 0.38129802 0.00794675 0.10182876 0.01257512
#  0.03701455 0.00608249 0.39666567]
# acc :  0.8990248611469087
# RF
# [2.78038490e-02 1.67750345e-04 1.30904989e-03 1.29391920e-03
#  2.68604832e-02 3.93559611e-01 5.61575589e-03 7.92068086e-02
#  2.40909018e-03 4.49881988e-03 1.41875340e-02 4.66685697e-03
#  4.38420472e-01]
# acc :  0.8535915332990583
# XGB
# [0.01311134 0.00178977 0.00865051 0.00337766 0.03526587 0.24189197
#  0.00975884 0.06960727 0.01454236 0.03254252 0.04658296 0.00757505
#  0.51530385]
# acc :  0.8902902185916939