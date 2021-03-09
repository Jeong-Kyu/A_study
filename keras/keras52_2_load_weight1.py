import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()


# plt.imshow(x_train[0], 'gray')
# plt.imshow(x_train[0])
# plt.show()
y_test1=y_test
# x_train = x_train.reshape(60000,28,28,1).astype('Float32')/255.
x_train = x_train.reshape(60000,28,28,1)/255.
x_test = x_test.reshape(10000,28,28,1)/255.
# x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

#OnehotEncoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = encoder.fit_transform(y_test.reshape(-1,1)).toarray()

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.2))
# model.add(Conv2D(???????))
model.add(Flatten())
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(10, activation='softmax'))

# model.save('../data/h5/k52_1_model1.h5')


##완성하기
#지표는 acc 0.985
#x_test 10개 y_pred 10개 출력
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
# k52_1_mnist_??? => k52_1_MCK.h5 이름을 바꿔줄것
# cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
# es = EarlyStopping(monitor='val loss', patience=5)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# hist = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.2, callbacks=[es, cp])

# model.save('../data/h5/k52_1_model2.h5')
# model.save_weights('../data/h5/k52_1_weight.h5')

# model1 = load_model('../data/h5/k52_1_model2.h5')

# result = model1.evaluate(x_test, y_test, batch_size=200)
# print('model1_loss : ', result[0])
# print('model1_accuracy : ', result[1])


model.load_weights ('../data/h5/k52_1_weight.h5')

result = model.evaluate(x_test, y_test, batch_size=200)
print('가중치_loss : ', result[0])
print('가중치_accuracy : ', result[1])


# 가중치_loss :  0.11399053037166595
# 가중치_accuracy :  0.9656000137329102

model2 = load_model('../data/h5/k52_1_model2.h5')
result2 = model2.evaluate(x_test, y_test, batch_size=200)
print('로드모델_loss : ', result2[0])
print('로드모델_accuracy : ', result2[1])

# 로드모델_loss :  0.11399053037166595
# 로드모델_accuracy :  0.9656000137329102