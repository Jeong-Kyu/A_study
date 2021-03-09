import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# 1. 데이터

# x, y = load_iris(return_X_y=True)
datasets = load_iris()
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

model =[LinearSVC, SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]
for a in range(5):
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

# LinearSVC              [1.         0.95833333 0.95833333 0.95833333 0.875     ]
# SVC                    [1.         0.95833333 0.95833333 0.95833333 1.        ]
# KNeighborsClassifier   [1.         1.         0.95833333 0.95833333 0.95833333]
# DecisionTreeClassifier [0.95833333 0.95833333 0.875      0.95833333 0.95833333]
# RandomForestClassifier [0.91666667 0.875      1.         0.95833333 0.95833333]

# scores :  [0.875      1.         0.95833333 0.95833333 0.91666667]
# scores :  [0.95833333 0.95833333 1.         0.95833333 0.91666667]
# scores :  [0.95833333 0.95833333 1.         0.91666667 1.        ]
# scores :  [0.91666667 0.91666667 0.95833333 1.         0.91666667]
# scores :  [0.95833333 0.95833333 1.         0.95833333 0.875     ]