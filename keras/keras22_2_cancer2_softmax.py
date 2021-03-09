# keras21_cancer1.py 를 다중분류 코딩하시오.

import numpy as np
from sklearn.datasets import load_breast_cancer

# 1. 데이터
datesets = load_breast_cancer()
# print(datesets.DESCR)
# print(datesets.feature_names)

x = datesets.data
y = datesets.target
# print(x.shape) #(569, 30)
# print(y.shape) #(569,)
# print(x[:5])
# print(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.utils import to_categorical


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train)
# print(y_train.shape)


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
model.add(Dense(100, activation='relu')) 
model.add(Dense(100, activation='relu')) 
model.add(Dense(10, activation='relu')) 
model.add(Dense(2, activation='softmax'))


# 3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, epochs=500, batch_size = 10, validation_split=0.2, callbacks = [early_stopping])

loss = model.evaluate(x_test, y_test)
print(loss)

y_pred=model.predict(x[-5:-1])

print(np.argmax(y_pred,axis=-1))
print(y[-5:-1])

# [0.5318993926048279, 0.9385964870452881]
# [[1. 0.]
#  [1. 0.]
#  [1. 0.]
#  [1. 0.]]
# [0 0 0 0]

#np.argmax 사용
# [0.48161518573760986, 0.9649122953414917]
# [0 0 0 0]
# [0 0 0 0]

# early_stopping 사용
# [0.04701797291636467, 0.9824561476707458]
# [0 0 0 0]
# [0 0 0 0]