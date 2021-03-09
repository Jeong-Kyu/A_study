import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import datetime
start1 = datetime.datetime.now()
# 1. 데이터

# x, y = load_iris(return_X_y=True)
# datasets = load_iris()
# x = datasets.data
# y = datasets.target

datasets = pd.read_csv('../data/csv/iris_sklean.csv', header=0, indax_col=0)
x = datasets.iloc[:,:-1]
y = datasets.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=66)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {'C': [1,10,100,1000], 'kernel':['linear']},
    {'C': [1,10,100], 'kernel':['rbf'], 'gamma':[0.001,0.0001]},
    {'C': [1,10,100,1000], 'kernel':['signodel'], 'gamma':[0.001,0.0001]}
]
# model =LinearSVC()
# model = GridSearchCV(SVC(), parameters, cv = kfold)
model = RandomizedSearchCV(SVC(), parameters, cv = kfold)

# scores = cross_val_score(model, x, y, cv=kfold)
# print("scores : ", scores)
# 3. 컴파일
model.fit(x_train,y_train)
# result = model.evaluate(x_test, y_test)
# result = model.score(x,y)
# print(result)
print('최적의 매개변수', model.best_estimator_)

y_pred=model.predict(x_test)
print('최종정답률', accuracy_score(y_test,y_pred))

aaa = model.score(x_test, y_test)
print(aaa)

end1 = datetime.datetime.now()
time_delta1 = end1 - start1
print('처리시간 : ', time_delta1)
# grid
# 최적의 매개변수 SVC(C=1, kernel='linear')
# 최종정답률 0.9666666666666667
# 0.9666666666666667
# 처리시간 :  0:00:00.088735
# random
# 최적의 매개변수 SVC(C=1, kernel='linear')
# 최종정답률 0.9666666666666667
# 0.9666666666666667
# 처리시간 :  0:00:00.051631