from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
# 1.
dataset = load_breast_cancer()
x = dataset.data
# x = pd.DataFrame(x, columns=dataset['feature_names'])
# x = x.drop(x.columns[[0,2,3,4,5,6,8,9,10,11,12,14,15,16,17,18,19,25,28,29]], axis='columns')

y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 44)
# 2.
# model = RandomForestClassifier(max_depth = 4)
model = GradientBoostingClassifier()
# 3.
model.fit(x_train, y_train)
# 4.
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print('acc : ', acc)

# [0.         0.00787229 0.96203388 0.03009382]
# acc :  0.9333333333333333

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
plot_feature_importances_dataset(model)
plt.show()
 
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
