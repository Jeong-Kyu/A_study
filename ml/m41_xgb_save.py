# early_stopping_rounds
# metric가 여러개면 마지막에 적용

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

# 1. 데이터
x, y = load_boston(return_X_y=True)
# datesets = load_boston()
# x = datesets.data
# y = datesets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle = True, random_state=66)

# 2. 모델
model = XGBRegressor(n_estimators = 1000, learning_rate = 0.01, n_jobs=8)

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'], eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=10)
# 각 훈련별 loss값이 반환
aaa = model.score(x_test, y_test)
# print("model.score : ", aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

# aaa :  0.9329663244922279
# r2  :  0.9329663244922279
# print("===========================")
# result = model.evals_result()
# print(result)

#저장
import pickle
# pickle.dump(model, open('../data/xgb_save/m39.pickle.dat','wb'))
import joblib
# joblib.dump(model,'../data/xgb_save/m39.joblib.dat')
# model.save_model('../data/xgb_save/m39.xgb.model')
# print('저장완료')

#불러오기
# model2 = pickle.load(open('../data/xgb_save/m39.pickle.dat','rb'))
# model2 = joblib.load('../data/xgb_save/m39.pickle.dat')
model2 = XGBRegressor()
model2.load_model('../data/xgb_save/m39.xgb.model')
print('불러오기')

r22 = model2.score(x_test, y_test)
print('r22 : ', r22)