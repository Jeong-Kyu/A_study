import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

kfold = KFold(n_splits=5, shuffle=True)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델

# model =LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

model =[LinearSVC, SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, LogisticRegression]
for a in range(6):
    scores = cross_val_score(model[a](), x_train, y_train, cv=kfold)
    print("scores : ", scores)

# 3. 컴파일

# model.fit(x,y)
# # result = model.evaluate(x_test, y_test)
# result = model.score(x,y)
# print(result)

# y_pred=model.predict(x)

# scores :  [0.82758621 0.68965517 0.92857143 0.67857143 0.89285714]
# scores :  [0.72413793 0.5862069  0.64285714 0.5        0.71428571]
# scores :  [0.72413793 0.62068966 0.67857143 0.64285714 0.78571429]
# scores :  [0.93103448 0.86206897 0.89285714 0.89285714 0.82142857]
# scores :  [0.93103448 1.         0.96428571 0.96428571 0.96428571]
# scores :  [0.96551724 0.96551724 0.89285714 0.92857143 0.92857143]