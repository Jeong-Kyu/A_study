# 완성하기

import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. 데이터 
dataset = load_wine()
x = dataset.data
y = dataset.target

#2. 모델링
model = RandomForestClassifier()

#3. 컴파일, 훈련
model.fit(x,y)
# result = model.evaluate(x_test, y_test)
result = model.score(x,y)
print(result)
y_pred=model.predict(x)
acc = accuracy_score(y, y_pred)
print("accuracy_score", acc)


# LinearSVC
# 0.7921348314606742
# accuracy_score 0.7921348314606742

# SVC
# 0.7078651685393258
# accuracy_score 0.7078651685393258

# KNeighborsClassifier
# 0.7865168539325843
# accuracy_score 0.7865168539325843

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
# 1.0