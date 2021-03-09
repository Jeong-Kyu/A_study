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
# 전처리 알아서 MinMaxScaler, train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66 )


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
'''
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)


'''
x_train = x_train.reshape(364,30,1,1)
x_test = x_test.reshape(114,30,1,1)
x_val = x_val.reshape(91,30,1,1)


#2. 모델링
model  = Sequential()
model.add(Conv2D(filters = 10,kernel_size=(1,1), strides=1, padding='same', input_shape = (30,1,1)))  # (input_dim * kernel_size + bias)*filter
#strides = 얼마나 건너서 자를건지 2 / (2,3)
model.add(MaxPool2D(pool_size=(1,1)))  # 2 / 3 / (2,3)
# model.add(Conv2D(9,(2,3)))
# model.add(Conv2D(8,2))
model.add(Flatten())
model.add(Dense(1))
model.add(Dense(1, activation='sigmoid')) #이진분류일 때 activation을 sigmoid로 사용
# linear -무한 ~ 무한 / relu 0 ~ 무한 / sigmoid 0 ~ 1

# 3. 컴파일, 훈련
                    # mean_squared_error                          #accuracy, mae
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# 이진분류일 때 loss를 binary_crossentropy로 사용  # metrics에 가급적 acc로 사용
# mse는 회귀모델
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, epochs=500, batch_size = 10, validation_data=(x_val, y_val), callbacks = [early_stopping])

loss = model.evaluate(x_test, y_test)
print(loss)

y_pred = model.predict_classes(x_test[-5:-1])
y_pred = np.transpose(y_pred)
print('y_pred : ', y_pred)
print('y값 : ', y_test[-5:-1])

# sigmoid 0~1 이므로 0.5 이상 = 1 미만 = 0 으로 설정
# 결과치 나오게 코딩!

# [0.18152697384357452, 0.9649122953414917]
# [[0.]
#  [0.]
#  [0.]
#  [0.]]
# [0 0 0 0]

#np.argmax 사용
# [0.648837149143219, 0.9385964870452881]
# [0 0 0 0]
# [0 0 0 0]

# EarlyStopping 사용
# [0.29604482650756836, 0.9736841917037964]
# [0 0 0 0]
# [0 0 0 0]

# 1, 0으로 값 나타내기
# [0.2550065815448761, 0.9473684430122375]
# WARNING:tensorflow:From c:\Study\keras\keras21_cancer1.py:52: Sequential.predict_classes (from tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.
# Instructions for updating:
# Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
# y_pred :  [[1 0 1 1]]
# y값 :  [1 0 1 1]

# cnn
# [0.24196220934391022, 0.9649122953414917]
# WARNING:tensorflow:From c:\Study\keras\keras41_cnn3_cancer.py:69: Sequential.predict_classes (from tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.
# Instructions for updating:
# Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
# y_pred :  [[1 0 1 1]]
# y값 :  [1 0 1 1]