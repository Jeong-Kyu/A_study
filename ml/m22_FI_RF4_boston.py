from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
# 1.
dataset = load_boston()
x = dataset.data
x = pd.DataFrame(x, columns=dataset['feature_names'])
x = x.drop(x.columns[[1,2,3,8,9]], axis='columns')

y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 44)
# 2.
model = RandomForestRegressor(max_depth = 4)
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

# [2.98930449e-02 2.97832405e-05 2.16380307e-03 8.27463734e-04
#  1.64582236e-02 3.94621188e-01 6.81119507e-03 8.36223929e-02
#  3.15479266e-03 2.24870636e-03 9.98767763e-03 5.48878561e-03
#  4.44692943e-01]
#  acc :  0.852641719178688
# [0.03526641 0.01976666 0.38158898 0.00793465 0.07313626 0.01751265
#  0.00372221 0.46107218]
# acc :  0.8471825190306405