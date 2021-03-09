# 실습 다:1 앙상블을 구현하시오
import numpy as np

#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200), range(100, 200)])
y1 = np.array([range(711, 811), range(1,101), range(100)])


# y2 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
#y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
# x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, shuffle=False, train_size=0.8, random_state=33)
# x2_train, x2_test, y1_train, y1_test = train_test_split(x2, y1, shuffle=False, train_size=0.8, random_state=33)
x1_train, x1_test, y1_train, y1_test, x2_train, x2_test = train_test_split(x1, y1, x2, shuffle=False, train_size=0.8, random_state=33)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)
#output1 = Dense(3)(dense1)

# 2
input2 = Input(shape=(4,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
#output2 = Dense(3)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(6)(middle1)
output = Dense(3)(middle1)
# 모델 분기
# 1
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)
# # 2
# output2 = Dense(10)(middle1)
# output2 = Dense(7)(output2)
# output2 = Dense(7)(output2)
# output2 = Dense(3)(output2)

# 모델 선언
model = Model(inputs = [input1, input2], outputs = output1)
model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam', metrics=['mae'])
model.fit([x1_train, x2_train], y1_train, epochs=10, batch_size=1, validation_split=0.2, verbose=1)

#4. 예측, 평가
loss = model.evaluate([x1_test, x2_test],y1_test, batch_size=1)

print("model.metrics_name : ", model.metrics_names)
print(loss)

y_predict= model.predict([x1_test, x2_test])


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y1_test, y_predict))
print("mse : ", mean_squared_error(y1_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y_predict)
print("R2 : ", r2)
