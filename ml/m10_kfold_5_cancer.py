import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
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

datasets = load_breast_cancer()
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

# test scores :  [0.9        0.93333333 1.         0.96666667 1.        ]
# val scores :  [0.95833333 1.         0.91666667 0.875      1.        ]

# 3. 컴파일
'''
model.fit(x,y)
# result = model.evaluate(x_test, y_test)
result = model.score(x,y)
print(result)

y_pred=model.predict(x)


# LinearSVC 0.9666666666666667
# SVC 0.9733333333333334
# KNeighborsClassifier 0.9666666666666667
# LogisticRegression 0.9733333333333334
# DecisionTreeClassifier 1.0
# RandomForestClassifier 1.0
# keras 1.0'''

# scores :  [0.92307692 0.85714286 0.91208791 0.93406593 0.96703297]
# scores :  [0.93406593 0.9010989  0.92307692 0.89010989 0.94505495]
# scores :  [0.96703297 0.93406593 0.94505495 0.91208791 0.89010989]
# scores :  [0.96703297 0.91208791 0.96703297 0.93406593 0.93406593]
# scores :  [0.93406593 0.95604396 0.95604396 0.94505495 0.97802198]
# scores :  [0.98901099 0.9010989  0.93406593 0.93406593 0.94505495]