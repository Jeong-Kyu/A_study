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
import datetime
start1 = datetime.datetime.now()

# 1. 데이터

# x, y = load_iris(return_X_y=True)
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

kfold = KFold(n_splits=5, shuffle=True)

#모델 : RandomForestClassifier

parameters = [
    {'n_estimators' : [100, 200],'max_depth' : [6, 8, 10, 12]},
    {'n_estimators' : [100, 200],'min_samples_leaf' : [3, 5, 7, 10]},
    {'n_estimators' : [100, 200],'min_samples_split' : [2, 3, 5, 10]},
    {'n_estimators' : [100, 200],'n_jobs' : [-1, 2, 4]}
]
# [
#     {'n_estimators' : [100, 200]},
#     {'max_depth' : [6, 8, 10, 12]},
#     {'min_samples_leaf' : [3, 5, 7, 10]},
#     {'min_samples_split' : [2, 3, 5, 10]},
#     {'n_jobs' : [-1, 2, 4]}
# ]
# 2. 모델

# model =LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold)
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
# 최적의 매개변수 :  RandomForestClassifier(max_depth=12)
# 최종정답률 : 0.011235955056179775
# scores :  [0. 0. 0. 0. 0.]
# 처리시간 :  0:04:20.621321
# random
# 최적의 매개변수 :  RandomForestClassifier(n_estimators=200)
# 최종정답률 : 0.011235955056179775
# scores :  [0.01408451 0.         0.         0.         0.        ]
# 처리시간 :  0:01:20.164271