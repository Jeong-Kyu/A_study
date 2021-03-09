from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier

# 1.
dataset = load_breast_cancer()
x = dataset.data
# x = pd.DataFrame(x, columns=dataset['feature_names'])
# x = x.drop(x.columns[[0,2,3,4,5,6,8,9,10,11,12,14,15,16,17,18,19,25,28,29]], axis='columns')

y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 44)

import datetime
start1 = datetime.datetime.now()

# 2.
# model = RandomForestClassifier(max_depth = 4)
model = GradientBoostingClassifier()
model = XGBClassifier(n_jobs=8)

# 3.
model.fit(x_train, y_train)
# 4.
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print('acc : ', acc)
end1 = datetime.datetime.now()
time_delta1 = end1 - start1
print('처리시간 : ', time_delta1)

# -1 처리시간 :  0:00:00.072782
#  8 처리시간 :  0:00:00.067707
# [0.         0.00787229 0.96203388 0.03009382]
# acc :  0.9333333333333333

# import matplotlib.pyplot as plt
# import numpy as np

# def plot_feature_importances_dataset(model):
#     n_features = dataset.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), dataset.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)
# plot_feature_importances_dataset(model)
# plt.show()
 
# [1.53861418e-06 9.44316408e-03 7.22960225e-04 5.52858176e-04
#  1.84196435e-05 3.86215014e-03 3.10997737e-04 4.07764341e-01
#  4.57534431e-04 1.66916493e-03 2.39024997e-03 2.06196777e-03
#  4.43542523e-04 9.23248664e-03 1.43122023e-03 1.38141394e-03
#  1.02470498e-03 1.06135585e-03 9.86048823e-04 3.25731893e-03
#  7.00442601e-02 6.12535675e-02 2.83194228e-01 4.77679963e-02
#  1.42655420e-02 2.23692306e-03 1.89495144e-02 5.36446801e-02
#  5.39697709e-04 3.01522163e-05]
# acc :  0.9736842105263158
# [0.01478615 0.41373791 0.01387478 0.06844593 0.05572643 0.27982999
#  0.05597593 0.01595215 0.0243598  0.05731093]
# acc :  0.9736842105263158

# XGB
# [4.9588699e-03 1.5423428e-02 6.3683779e-04 2.2023508e-02 7.9919444e-03
#  9.5059077e-04 1.1493003e-02 1.5286781e-01 3.9727203e-04 2.7880725e-03
#  3.4555981e-03 5.6501115e-03 2.8066258e-03 3.8451280e-03 2.8832396e-03
#  3.3951809e-03 3.4120989e-03 5.8727746e-04 9.2994876e-04 3.7213673e-03
#  1.1663227e-01 2.0733794e-02 5.1057857e-01 1.3440680e-02 6.8047098e-03
#  0.0000000e+00 2.1567145e-02 5.5507991e-02 1.3584920e-03 3.1585069e-03]
# acc :  0.9824561403508771