

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import datetime
start1 = datetime.datetime.now()
from xgboost import XGBClassifier,plot_importance
# 1. 데이터

# x, y = load_iris(return_X_y=True)
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

kfold = KFold(n_splits=7, shuffle=True)

#모델 : RandomForestClassifier
#     {'n_estimators' : [100, 200]},
#     {'max_depth' : [6, 8, 10, 12]},
#     {'min_samples_leaf' : [3, 5, 7, 10]},
#     {'min_samples_split' : [2, 3, 5, 10]},
#     {'n_jobs' : [-1, 2, 4]}
# ]
#     {'C': [1,10,100,1000], 'kernel':['linear']},
#     {'C': [1,10,100], 'kernel':['rbf'], 'gamma':[0.001,0.0001]},
#     {'C': [1,10,100,1000], 'kernel':['signodel'], 'gamma':[0.001,0.0001]}
parameters = [
    {'n_estimators' : [100, 200, 300], 'learing_rate' : [0.1,0.3,0.001,0.01],
    'max_depth' : [4,5,6]},
    {'n_estimators' : [90, 100, 110], 'learing_rate' : [0.1,0.001,0.01],
    'max_depth' : [4,5,6], 'colsample_bytree' : [0.6, 0.9, 1]},
    {'n_estimators' : [90, 110], 'learing_rate' : [0.1,0.001,0.5],
    'max_depth' : [4,5,6], 'colsample_bytree' : [0.6, 0.9, 1], 'colsample_bylevel' : [0.6,0.7,0.9]}
]
# 2. 모델

# model =LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()

model = GridSearchCV(XGBClassifier(), parameters, cv = kfold, verbose=1)
scores = cross_val_score(model, x_train, y_train, cv=kfold)
model.fit(x_train,y_train)
print('최적의 매개변수 : ', model.best_estimator_)
y_pred = model.predict(x_test)
print('최종정답률 :', accuracy_score(y_test, y_pred))
print("scores : ", scores)

end1 = datetime.datetime.now()
time_delta1 = end1 - start1
print('처리시간 : ', time_delta1)
# grid
# 최적의 매개변수 :  RandomForestClassifier(max_depth=10)
# 최종정답률 : 0.9333333333333333
# scores :  [1.         0.91666667 0.95833333 0.91666667 1.        ]
# 처리시간 :  0:02:42.087172

import matplotlib.pyplot as plt
import numpy as np

plot_importance(model)
plt.show()

# 최종정답률 : 0.9
# scores :  [1.         1.         1.         0.88235294 0.94117647 0.76470588
#  1.        ]
# 처리시간 :  0:10:43.962886