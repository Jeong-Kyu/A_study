from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
# 1.
dataset = load_breast_cancer()
x = dataset.data
x = pd.DataFrame(x, columns=dataset['feature_names'])
x = x.drop(x.columns[[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,26,28,29]], axis='columns')

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

# [0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.00677572 0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.05612587 0.78000877 0.01008994
#  0.00995429 0.00677572 0.         0.13026968 0.         0.        ]
# acc :  0.9385964912280702

# [0.         0.05612587 0.78678449 0.01008994 0.01673002 0.
#  0.13026968]
# acc :  0.9385964912280702