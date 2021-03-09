from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
# 1.
dataset = load_breast_cancer()
x = dataset.data
x = pd.DataFrame(x, columns=dataset['feature_names'])
x = x.drop(x.columns[[10,11,12,13,14,15,18,20,21,25]], axis='columns')

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

# [0.04330929 0.01350546 0.04347002 0.06275319 0.00420575 0.00791252
#  0.05983544 0.08927697 0.00167343 0.00082558 0.01154935 0.0034987
#  0.00921501 0.04506277 0.00178787 0.00417465 0.00260467 0.00350404
#  0.00103134 0.00369216 0.13234182 0.01762801 0.1094531  0.10499581
#  0.01435919 0.01220798 0.04155619 0.13531661 0.01344447 0.0058086 ]
#  acc :  0.9649122807017544

# [0.05940356 0.02056662 0.05767119 0.0508774  0.00488509 0.01547043
#  0.0402173  0.15425028 0.00280791 0.00349482 0.01745632 0.00708995
#  0.00535363 0.14125574 0.22299594 0.02250466 0.05447247 0.10427089
#  0.00848181 0.00647402]
#  acc :  0.9649122807017544