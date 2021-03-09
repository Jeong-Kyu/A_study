import numpy as np

from sklearn.datasets import load_wine

dataset = load_wine()


x = dataset.data
y = dataset.target

# print(x)
# print(y)
# print(x.shape) #(178, 13)
# print(y.shape) #(178, )

# 실습 DNN 완성

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66 )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x = scaler.transform(x)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 2차원 데이터로 변환하기
y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
y_val = y_val.reshape(-1,1)

# print(y_test)
# print(y_train)
# print(y_val)


# 원-핫 인코딩 적용
encoder = OneHotEncoder()
encoder.fit(y_test)
y_test = encoder.transform(y_test)
encoder.fit(y_train)
y_train = encoder.transform(y_train)
encoder.fit(y_val)
y_val = encoder.transform(y_val)

y_test=y_test.toarray()
y_train=y_train.toarray()
y_val=y_val.toarray()
'''
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)

# 2. 모델
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,Flatten
x = x.reshape(178,13,1,1)
x_train = x_train.reshape(113,13,1,1)
x_test = x_test.reshape(36,13,1,1)
x_val = x_val.reshape(29,13,1,1)


#2. 모델링
model  = Sequential()
model.add(Conv2D(filters = 10,kernel_size=(1,1), strides=1, padding='same', input_shape = (13,1,1)))  # (input_dim * kernel_size + bias)*filter
#strides = 얼마나 건너서 자를건지 2 / (2,3)
model.add(MaxPool2D(pool_size=(1,1)))  # 2 / 3 / (2,3)
# model.add(Conv2D(9,(2,3)))
# model.add(Conv2D(8,2))
model.add(Flatten())
model.add(Dense(1))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, epochs=500, batch_size = 10, validation_data=(x_val, y_val), callbacks = [early_stopping])

loss = model.evaluate(x_test, y_test)
print(loss)
# y_pred=model.predict(x)
# print(y_pred)

y_pred=model.predict(x)
print(np.argmax(y_pred,axis=-1))
print(y)

# [0.002698038937523961, 1.0]'''

# cnn
# [0.6162065863609314, 0.8611111044883728]