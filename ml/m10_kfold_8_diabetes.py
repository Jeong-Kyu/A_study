import numpy as np
from sklearn.datasets import load_diabetes
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

datasets = load_diabetes()
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

model =[LinearSVC, SVC, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor]
for a in range(5):
    scores = cross_val_score(model[a](), x_train, y_train, cv=kfold)
    print("scores : ", scores)

# 3. 컴파일

# model.fit(x,y)
# # result = model.evaluate(x_test, y_test)
# result = model.score(x,y)
# print(result)

# y_pred=model.predict(x)

# scores :  [0.         0.         0.01408451 0.         0.01428571]
# scores :  [0.         0.01408451 0.         0.01428571 0.        ]
# scores :  [0.49343214 0.31078701 0.3234071  0.31592199 0.44172257]
# scores :  [ 0.10312456 -0.03404684  0.0531885  -0.35575244 -0.37367122]
# scores :  [0.35710107 0.45188313 0.41210229 0.50907712 0.5401425 ]