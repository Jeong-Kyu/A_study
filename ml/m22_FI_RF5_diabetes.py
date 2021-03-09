from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
# 1.
dataset = load_diabetes()
x = dataset.data
x = pd.DataFrame(x, columns=dataset['feature_names'])
x = x.drop(x.columns[[1,3,5,7]], axis='columns')

y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 44)
# 2.
model = RandomForestClassifier(max_depth = 4)
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

# [0.0982035  0.00944252 0.125621   0.11018434 0.11614287 0.10706286
#  0.11506832 0.07715289 0.12025175 0.12086996]
# acc :  0.0
# [0.14478263 0.17663638 0.16863402 0.17178241 0.17188971 0.16627485]
# acc :  0.011235955056179775


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
# print(cut_columns(model.feature_importances_,datasets.feature_names,8))