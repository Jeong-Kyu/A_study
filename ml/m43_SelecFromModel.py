from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score
import numpy as np

x,y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state = 66)

model = XGBRegressor(n_jobs=8)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("R2 : ",score)

thresholds = np.sort(model.feature_importances_)
print(thresholds)
print(np.sum(thresholds))

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) #thresh값 이상의 것을 전부 처리     디폴트 prefit = False
    select_x_trian = selection.transform(x_train)
    print(select_x_trian.shape)

    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_trian, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d R2: %.2f%%" %(thresh, select_x_trian.shape[1], score*100))  #부스트 트리 계열에서 사용

# print(model.coef_)
# print(model.intercept_)
# AttributeError: Coefficients are not defined for Booster type None
