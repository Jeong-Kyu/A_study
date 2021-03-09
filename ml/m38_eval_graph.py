# early_stopping_rounds

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)
# datesets = load_boston()
# x = datesets.data
# y = datesets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle = True, random_state=66)

# 2. 모델
model = XGBClassifier(n_estimators = 1000, learning_rate = 0.01, n_jobs=8)

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['logloss','error'], eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=10)
# 각 훈련별 loss값이 반환
aaa = model.score(x_test, y_test)
print("aaa : ", aaa)

y_pred = model.predict(x_test)
accuracy_score = accuracy_score(y_test, y_pred)
print("accuracy_score : ", accuracy_score)

# aaa :  0.9329663244922279
# r2  :  0.9329663244922279
print("===========================")
results = model.evals_result()
# print(results)

import matplotlib.pyplot as plt

epochs = len(results['validation_0']['logloss'])
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax. legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
# plt.show()

fig, ac = plt.subplots()
ac.plot(x_axis, results['validation_0']['error'], label='Train')
ac.plot(x_axis, results['validation_1']['error'], label='Test')
ac. legend()
plt.ylabel('error')
plt.title('XGBoost error')
plt.show()
