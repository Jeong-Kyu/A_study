# 완성하기

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#1. 데이터 
dataset = load_boston()
x = dataset.data
y = dataset.target

#1_2. 데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x = scaler.transform(x)
#2. 모델링
# model = LinearRegression()
model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()


#3. 컴파일, 훈련
model.fit(x_train,y_train)
# result = model.evaluate(x_test, y_test)
result = model.score(x_test,y_test)
print(result)
y_pred=model.predict(x_test)
R2 = r2_score(y_test, y_pred)
print("r2_score", R2)


# LinearRegression
# 0.8005220851783466
# r2_score 0.8005220851783466

# KNeighborsRegressor()
# 0.8265307833211177
# r2_score 0.8265307833211177

# DecisionTreeRegressor
# 0.7795837307971942
# r2_score 0.7795837307971942

# RandomForestRegressor
# 0.9230836530623246
# r2_score 0.9230836530623246

# keras
# # R2 :  0.978892450054018