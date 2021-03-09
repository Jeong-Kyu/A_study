# eval을 이용한 검증 (eval_mertuc, eval_set, eval_result)

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

# 1. 데이터
x, y = load_wine(return_X_y=True)
# datesets = load_wine()
# x = datesets.data
# y = datesets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle = True, random_state=66)

# 2. 모델
model = XGBClassifier(n_estimators = 100, learning_rate = 0.01, n_jobs=8)

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['mlogloss','merror'], eval_set=[(x_train, y_train), (x_test, y_test)])
# 각 훈련별 loss값이 반환
aaa = model.score(x_test, y_test)
print("aaa : ", aaa)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy : ", accuracy)

# aaa :  0.9722222222222222
# accuracy :  0.9722222222222222
print("===========================")
result = model.evals_result()
print(result)