import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# 2. 모델

model =LogisticRegression()


# 3. 컴파일

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# 다중분류에서 categorical_crossentropy를 사용

# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

# model.fit(x_train, y_train, epochs=500, batch_size = 10, validation_split=0.2, callbacks = [early_stopping])
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
# keras 1.0