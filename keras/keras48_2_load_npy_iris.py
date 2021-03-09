import numpy as np

x_data = np.load('./data/iris_x.npy')
y_data = np.load('./data/iris_y.npy')

print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8, shuffle = True, random_state = 66 )
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True, random_state = 66 )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_data = scaler.transform(x_data)

#원핫 인코딩 OneHotEncoding
# from tensorflow.keras.utils import to_categorical
# # from keras.utils.np_utils import to_categorical

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train)
# print(y_train.shape) # (120,3) test (30,3)  스칼라에서 벡터로 변환시킴

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

encoder = OneHotEncoder()
y_test = encoder.fit_transform(y_test.reshape(-1,1)).toarray()
y_train = encoder.fit_transform(y_train.reshape(-1,1)).toarray()
y_val = encoder.fit_transform(y_val.reshape(-1,1)).toarray()
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)

# 2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,Flatten
x_data = x_data.reshape(150,4,1,1)
x_train = x_train.reshape(96,4,1,1)
x_test = x_test.reshape(30,4,1,1)
x_val = x_val.reshape(24,4,1,1)


#2. 모델링
model  = Sequential()
model.add(Conv2D(filters = 10,kernel_size=(1,1), strides=1, padding='same', input_shape = (4,1,1)))  # (input_dim * kernel_size + bias)*filter
#strides = 얼마나 건너서 자를건지 2 / (2,3)
model.add(MaxPool2D(pool_size=(1,1)))  # 2 / 3 / (2,3)
# model.add(Conv2D(9,(2,3)))
# model.add(Conv2D(8,2))
model.add(Flatten())
model.add(Dense(1))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# 다중분류에서 categorical_crossentropy를 사용

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k46_7_iris_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

hist = model.fit(x_train, y_train, epochs=500, batch_size = 10, validation_data=(x_val, y_val), callbacks = [early_stopping, cp])

loss = model.evaluate(x_test, y_test)
print(loss)