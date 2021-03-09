from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
# 1.
dataset = load_wine()
x = dataset.data
x = pd.DataFrame(x, columns=dataset['feature_names'])
x = x.drop(x.columns[[2,3,4,7,8]], axis='columns')

y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 44)
# 2.
model = DecisionTreeClassifier(max_depth = 4)
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

# [0.         0.01723824 0.         0.         0.         0.
#  0.17751337 0.         0.         0.         0.05577403 0.32933594
#  0.42013842]
# acc :  0.8888888888888888
# [0.0179565  0.         0.         0.17679511 0.         0.05577403
#  0.32933594 0.42013842]
# acc :  0.8888888888888888