# 완성하기

import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. 데이터 
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

#1_2. 데이터 전처리
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size= 0.8, shuffle = True, random_state = 66, )

# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)
#2. 모델링
model = LogisticRegression()

#3. 컴파일, 훈련
model.fit(x,y)
# result = model.evaluate(x_test, y_test)
result = model.score(x,y)
print(result)
y_pred=model.predict(x)
acc = accuracy_score(y, y_pred)
print("accuracy_score", acc)


# LinearSVC
# 0.9244288224956063
# accuracy_score 0.9244288224956063

# SVC
# 0.9226713532513181
# accuracy_score 0.9226713532513181

# KNeighborsClassifier
# 0.9472759226713533
# accuracy_score 0.9472759226713533

# LogisticRegression
# 0.9437609841827768
# accuracy_score 0.9437609841827768
# DecisionTreeClassifier
# 1.0
# accuracy_score 1.0

# RandomForestClassifier
# 1.0
# accuracy_score 1.0

# keras
# 0.9736841917037964