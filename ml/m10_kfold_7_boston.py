import numpy as np
from sklearn.datasets import load_boston
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

datasets = load_boston()
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

# scores :  [nan nan nan nan nan]
# scores :  [nan nan nan nan nan]
# scores :  [0.46815965 0.31640056 0.52149175 0.47102217 0.55755859]
# scores :  [0.58444086 0.44628436 0.76767295 0.77809541 0.85494046]
# scores :  [0.82444839 0.85445964 0.85708406 0.8741382  0.81139243]