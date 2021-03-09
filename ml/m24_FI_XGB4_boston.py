from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

# 1.
dataset = load_boston()
x = dataset.data
x = pd.DataFrame(x, columns=dataset['feature_names'])
x = x.drop(x.columns[[1,2,3,8]], axis='columns')

y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 44)

import datetime
start1 = datetime.datetime.now()
end1 = datetime.datetime.now()
time_delta1 = end1 - start1
print('처리시간 : ', time_delta1)
# 2.
# model = RandomForestClassifier(max_depth = 4)
model = GradientBoostingRegressor()
model = XGBRegressor(n_jobs=-1)

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
 
# [0.02923112 0.00041448 0.0027793  0.00113228 0.02952698 0.3796114 
#  0.00862218 0.10018367 0.00073107 0.01156829 0.03277563 0.00728964
#  0.39613397]
# acc :  0.8932927259768633
# [0.02633349 0.03025513 0.38129802 0.00794675 0.10182876 0.01257512
#  0.03701455 0.00608249 0.39666567]
# acc :  0.8990248611469087