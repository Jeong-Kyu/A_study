import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
(x_train, y_train),(x_test, y_test) = mnist.load_data()


x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

x = x.reshape(70000,28*28)
# print(x.shape) (70000, 28, 28)

pca =PCA(n_components = 154)
x = pca.fit_transform(x)
# print(x.shape) (70000, 154)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 44)

# x_train = x_train.reshape(56000,154)/255.
# x_test = x_test.reshape(14000,154)/255.
# x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

#OnehotEncoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = encoder.fit_transform(y_test.reshape(-1,1)).toarray()
print(y.shape)
print(x_train[1])
print(y_train[1])
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Dense(30,input_shape=(154,), activation='relu'))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=128, epochs=100)

loss, acc = model.evaluate(x_test, y_test, batch_size=200)
print('loss : ', loss)
print('acc : ', acc)

# cnn
# loss :  0.1353285312652588
# acc :  0.9728000164031982
# [7 2 1 0 4 1 4 9 5 9]
# [7 2 1 0 4 1 4 9 6 9]

# dnn
# loss :  0.336788147687912
# acc :  0.9638000130653381
# [7 2 1 0 4 1 4 9 5 9]
# [7 2 1 0 4 1 4 9 5 9]


# loss :  0.46745622158050537
# acc :  0.9564999938011169