from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier, plot_importance
# 1.
dataset = load_iris()
x = dataset.data
x = pd.DataFrame(x, columns=dataset['feature_names'])
x = x.drop(x.columns[[0,1]], axis='columns')

y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 44)

import datetime
start1 = datetime.datetime.now()

# 2.
# model = RandomForestClassifier(max_depth = 4)
# model = GradientBoostingClassifier()
model = XGBClassifier(n_jobs=8)
# 3.
model.fit(x_train, y_train)
# 4.
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print('acc : ', acc)

# end1 = datetime.datetime.now()
# time_delta1 = end1 - start1
# print('처리시간 : ', time_delta1)

#  -1 처리시간 :  0:00:00.075796
#  8  처리시간 :  0:00:00.074176
# [0.         0.00787229 0.96203388 0.03009382]
# acc :  0.9333333333333333

import matplotlib.pyplot as plt
import numpy as np

# def cut_columns(feature_importances,columns,number):
#     temp = []
#     for i in feature_importances:
#         temp.append(i)
#     temp.sort()
#     temp=temp[:number]
#     result = []
#     for j in temp:
#         index = feature_importances.tolist().index(j)
#         result.append(columns[index])
#     return result
# print(cut_columns(model.feature_importances_,dataset.feature_names,1))
'''
def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
plot_feature_importances_dataset(model)
'''
plot_importance(model)
plt.show()
 
# [0.00610507 0.01239617 0.67175367 0.30974509]
# acc :  0.9666666666666667
# [0.61598008 0.38401992]
# acc :  0.9666666666666667
