import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터

# x, y = load_iris(return_X_y=True)
datasets = load_iris()
x = datasets.data
y = datasets.target

# print(datasets.DESCR)
# print(datasets.feature_names)

# print(x.shape) # (150,4)
# print(y.shape) # (150, )
# print(x[:5])
# print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#원핫 인코딩 OneHotEncoding
from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)
print(y_train.shape) # (120,3) test (30,3)  스칼라에서 벡터로 변환시킴

# 2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))
# softmax(다중분류에서 사용) 분류 경우의 수만큼 output 노드를 잡음
# 분리된 값의 합은 1 
# 가장 큰 값을 선택

# 3. 컴파일

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# 다중분류에서 categorical_crossentropy를 사용

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, epochs=500, batch_size = 10, validation_split=0.2, callbacks = [early_stopping])

loss = model.evaluate(x_test, y_test)
print(loss)

y_pred=model.predict(x[-5:-1])
print(np.argmax(y_pred,axis=-1))
print(y[-5:-1])


# oh_encoder = OneHotEncoder()
# oh_encoder.fit(labels)
# oh_labels = oh_encoder.transform(labels)

# to_categorical : 무조건 0부터 시작



# [0.0731370747089386, 0.9666666388511658]
# [[0.0000000e+00 1.4650373e-32 1.0000000e+00]
#  [0.0000000e+00 4.3426609e-29 1.0000000e+00]
#  [0.0000000e+00 2.2736888e-30 1.0000000e+00]
#  [0.0000000e+00 2.6349163e-33 1.0000000e+00]]
# [2 2 2 2]

#np.argmax 사용
# [0.06896660476922989, 1.0]
# [2 2 2 2]
# [2 2 2 2]

# EarlyStopping 사용
# [0.09917701035737991, 0.9666666388511658]
# [2 2 2 2]
# [2 2 2 2]
