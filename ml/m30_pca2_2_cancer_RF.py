import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

datasets = load_breast_cancer()
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

pca =PCA(n_components = 29)
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

# [0.         0.00787229 0.96203388 0.03009382]
# acc :  0.9333333333333333
# RF
# [0.03900304 0.01842087 0.05459043 0.04686347 0.00711849 0.01264556
#  0.06176893 0.11303675 0.00129019 0.00282914 0.00409799 0.00276916
#  0.00876777 0.01855542 0.00329545 0.0045784  0.00840925 0.0014817
#  0.00250567 0.00300812 0.08799284 0.0137395  0.15620285 0.13981192
#  0.01830764 0.01181742 0.03207978 0.11054159 0.00674902 0.00772164]
# acc :  0.9649122807017544
# XGB
# [4.9588699e-03 1.5423428e-02 6.3683779e-04 2.2023508e-02 7.9919444e-03
#  9.5059077e-04 1.1493003e-02 1.5286781e-01 3.9727203e-04 2.7880725e-03
#  3.4555981e-03 5.6501115e-03 2.8066258e-03 3.8451280e-03 2.8832396e-03
#  3.3951809e-03 3.4120989e-03 5.8727746e-04 9.2994876e-04 3.7213673e-03
#  1.1663227e-01 2.0733794e-02 5.1057857e-01 1.3440680e-02 6.8047098e-03
#  0.0000000e+00 2.1567145e-02 5.5507991e-02 1.3584920e-03 3.1585069e-03]
# acc :  0.9824561403508771