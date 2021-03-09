from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
# 1.
dataset = load_wine()
x = dataset.data
x = pd.DataFrame(x, columns=dataset['feature_names'])
x = x.drop(x.columns[[3,5,7,8]], axis='columns')

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

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
plot_feature_importances_dataset(model)
plt.show()
 
# [6.20927471e-02 4.02905163e-02 1.19621922e-02 9.44878329e-06
#  7.96221009e-03 1.57896193e-08 1.96477173e-01 1.55867109e-03
#  1.47329467e-03 2.18116160e-01 2.01217916e-02 1.49338501e-01
#  2.90597278e-01]
# acc :  0.9166666666666666
# [0.06207569 0.0322889  0.01513384 0.00977241 0.21456361 0.23026225
#  0.02093538 0.12288826 0.29207966]
# acc :  0.9166666666666666