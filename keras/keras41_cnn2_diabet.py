#실습 : 19_1, 2, 3, 4, 5, EarlyStopping까지 총 6개의 파일을 완성하시오.

import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D,MaxPool2D, Flatten

#1. 데이터 
dataset = load_diabetes()
x = dataset.data
y = dataset.target

#print(dataset.DESCR)

#x = x / np.max(x)
#print(np.max(x), np.min(x)) # 정규화

#1_2. 데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size= 0.8, shuffle = True, random_state = 66, )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
'''
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)


'''
x_train = x_train.reshape(282,10,1,1)
x_test = x_test.reshape(89,10,1,1)
x_val = x_val.reshape(71,10,1,1)


#2. 모델링
model  = Sequential()
model.add(Conv2D(filters = 10,kernel_size=(1,1), strides=1, padding='same', input_shape = (10,1,1)))  # (input_dim * kernel_size + bias)*filter
#strides = 얼마나 건너서 자를건지 2 / (2,3)
model.add(MaxPool2D(pool_size=(1,1)))  # 2 / 3 / (2,3)
# model.add(Conv2D(9,(2,3)))
# model.add(Conv2D(8,2))
model.add(Flatten())
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
'''
EarlyStopping
'''
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=7, validation_data= (x_val, y_val), callbacks = [early_stopping])

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
# loss :  3379.458984375
# mae :  47.35618591308594
# RMSE :  58.13311275393621
# R2 :  0.47928539874511966

#데이터 x를 전처리한 후
# loss :  3291.452880859375
# mae :  46.496116638183594
# RMSE :  57.37118551454562
# R2 :  0.49284554101046385

#데이터 x_train잡아서 전처리한 후....
# loss :  3421.5537109375
# mae :  47.82155227661133
# RMSE :  58.49405010020266
# R2 :  0.47279929140593635

#발리데이션 test분리
# loss :  3369.262451171875
# mae :  48.33604431152344
# RMSE :  58.04534944194592
# R2 :  0.5128401315682825

#Earlystopping 적용
# loss :  57.708213806152344
# mae :  5.144794464111328
# RMSE :  7.596591897809446
# R2 :  0.991656001135139


# loss :  6162.50048828125
# mae :  62.28391647338867
# RMSE :  78.50159664503691
# R2 :  0.05046805612688887'''

# cnn
# loss :  3194.252197265625
# mae :  46.7601318359375
# RMSE :  56.517714046839565
# R2 :  0.507822478023054