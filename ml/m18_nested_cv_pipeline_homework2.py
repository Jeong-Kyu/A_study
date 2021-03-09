# RandomForest
# pipeline으로 25번 돌리기
# 데이터 wine

# RandomForest
# pipeline으로 25번 돌리기
# 데이터 diabetes

# RandomSearch, GridSearch -- Pipeline
# RandomForest

import numpy as np
from sklearn.datasets import load_diabetes, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC, SVC
import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터

# x, y = load_iris(return_X_y=True)
datasets = load_wine()
x = datasets.data
y = datasets.target
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=66)

kfold = KFold(n_splits=3, shuffle=True)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2.
# pipeline -> 전처리까지 합치기
# model = Pipeline([("scaler", MinMaxScaler()),("mal", SVC())])
scalers = [MinMaxScaler, StandardScaler]

for a in scalers:
    parameters1 =[
    {'randomforestclassifier__n_estimators' : [100, 200],'randomforestclassifier__max_depth' : [6, 8, 10, 12]},
    {'randomforestclassifier__n_estimators' : [100, 200],'randomforestclassifier__min_samples_leaf' : [3, 5, 7, 10]},
    {'randomforestclassifier__n_estimators' : [100, 200],'randomforestclassifier__min_samples_split' : [2, 3, 5, 10]},
    {'randomforestclassifier__n_estimators' : [100, 200],'randomforestclassifier__n_jobs' : [-1, 2, 4]}
    ]
    pipe = make_pipeline(a(), RandomForestClassifier())
    # pipe = Pipeline([("scaler", MinMaxScaler()),("mal", RandomForestClassifier())])
    model = GridSearchCV(pipe, parameters1, cv =kfold)

    print(model)
    score = cross_val_score(model, x, y, cv=kfold)
    print(score)
    '''
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print("GridSearch : ", result)

for a in scalers:
    parameters2 = [
    {'mal__n_estimators' : [100, 200],'mal__max_depth' : [6, 8, 10, 12]},
    {'mal__n_estimators' : [100, 200],'mal__min_samples_leaf' : [3, 5, 7, 10]},
    {'mal__n_estimators' : [100, 200],'mal__min_samples_split' : [2, 3, 5, 10]},
    {'mal__n_estimators' : [100, 200],'mal__n_jobs' : [-1, 2, 4]}
    ]
    # pipe = make_pipeline(a(), RandomForestClassifier())
    pipe = Pipeline([("scaler", MinMaxScaler()),("mal", RandomForestClassifier())])
    model = RandomizedSearchCV(pipe, parameters2, cv = 5)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print("RandomSearch : ", result)
    
# GridSearch :  0.011235955056179775
# GridSearch :  0.02247191011235955
# RandomSearch :  0.02247191011235955
# RandomSearch :  0.011235955056179775'''


# GridSearchCV(cv=KFold(n_splits=3, random_state=None, shuffle=True),
#              estimator=Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
#                                        ('randomforestclassifier',
#                                         RandomForestClassifier())]),
#              param_grid=[{'randomforestclassifier__max_depth': [6, 8, 10, 12],
#                           'randomforestclassifier__n_estimators': [100, 200]},
#                          {'randomforestclassifier__min_samples_leaf': [3, 5, 7,
#                                                                        10],
#                           'randomforestclassifier__n_estimators': [100, 200]},
#                          {'randomforestclassifier__min_samples_split': [2, 3, 5,
#                                                                         10],
#                           'randomforestclassifier__n_estimators': [100, 200]},
#                          {'randomforestclassifier__n_estimators': [100, 200],
#                           'randomforestclassifier__n_jobs': [-1, 2, 4]}])
# [0.95       0.98305085 1.        ]
# GridSearchCV(cv=KFold(n_splits=3, random_state=None, shuffle=True),
#              estimator=Pipeline(steps=[('standardscaler', StandardScaler()),
#                                        ('randomforestclassifier',
#                                         RandomForestClassifier())]),
#              param_grid=[{'randomforestclassifier__max_depth': [6, 8, 10, 12],
#                           'randomforestclassifier__n_estimators': [100, 200]},
#                          {'randomforestclassifier__min_samples_leaf': [3, 5, 7,
#                                                                        10],
#                           'randomforestclassifier__n_estimators': [100, 200]},
#                          {'randomforestclassifier__min_samples_split': [2, 3, 5,
#                                                                         10],
#                           'randomforestclassifier__n_estimators': [100, 200]},
#                          {'randomforestclassifier__n_estimators': [100, 200],
#                           'randomforestclassifier__n_jobs': [-1, 2, 4]}])
# [0.98333333 1.         0.94915254]