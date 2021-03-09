from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier

# 1.
dataset = load_diabetes()
x = dataset.data
x = pd.DataFrame(x, columns=dataset['feature_names'])
x = x.drop(x.columns[[1,7]], axis='columns')

y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 44)

import datetime
start1 = datetime.datetime.now()
end1 = datetime.datetime.now()
time_delta1 = end1 - start1
print('처리시간 : ', time_delta1)
# 2.
# model = RandomForestClassifier(max_depth = 4)
model = GradientBoostingClassifier()
model = XGBClassifier(n_jobs=-1)

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
 
# [0.09943162 0.01250567 0.13017023 0.0897461  0.09800625 0.1622884
#  0.12445862 0.04798841 0.11897218 0.11643251]
# acc :  0.0
# [0.10106657 0.15215195 0.09590524 0.1063308  0.14627536 0.12786762
#  0.14037334 0.13002911]
# acc :  0.0