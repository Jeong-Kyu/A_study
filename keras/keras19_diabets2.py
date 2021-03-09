#실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지 총 6개의 파일을 완성하시오.

import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#1. 데이터 
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])
print(x.shape, y.shape) #(442, 10) (442,)
print(np.max(x), np.min(x))
print(dataset.feature_names)
#print(dataset.DESCR)

x = x / np.max(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

#2. 모델링
model = Sequential()
model.add(Dense(100, input_dim=10, activation = 'relu')) # 기본값 : activation='linear' 
model.add(Dense(80, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs = 1000, batch_size = 7, validation_split = 0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)

#RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""
**Data Set Characteristics:**
  :Number of Instances: 442
  :Number of Attributes: First 10 columns are numeric predictive values
  :Target: Column 11 is a quantitative measure of disease progression one year after baseline
  :Attribute Information:
      - age     age in years
      - sex
      - bmi     body mass index
      - bp      average blood pressure
      - s1      tc, T-Cells (a type of white blood cells)
      - s2      ldl, low-density lipoproteins
      - s3      hdl, high-density lipoproteins
      - s4      tch, thyroid stimulating hormone
      - s5      ltg, lamotrigine
      - s6      glu, blood sugar level
"""

#데이터 전처리 전
# loss :  3317.64599609375
# mae :  47.06387710571289
# RMSE :  57.59901189718801
# R2 :  0.488809627121195

#데이터 엉망 처리 후
# loss :  6436.35888671875
# mae :  61.487239837646484
# RMSE :  80.2269237011168
# R2 :  0.008271306354807884