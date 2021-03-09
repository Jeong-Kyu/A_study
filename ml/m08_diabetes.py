# 완성하기

import numpy as np
from sklearn.datasets import load_diabetes
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
dataset = load_diabetes()
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
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()


#3. 컴파일, 훈련
model.fit(x_train,y_train)
# result = model.evaluate(x_test, y_test)
result = model.score(x_test,y_test)
print(result)
y_pred=model.predict(x_test)
R2 = r2_score(y_test, y_pred)
print("r2_score", R2)


# LinearRegression
# 0.5063891053505036
# r2_score 0.5063891053505036

# KNeighborsRegressor()
# 0.3741821819765594
# r2_score 0.3741821819765594

# DecisionTreeRegressor
# -0.08695502475217198
# r2_score -0.08695502475217198

# RandomForestRegressor
# 0.39146056856173617
# r2_score 0.39146056856173617

# keras
# R2 :  0.9979341930647975