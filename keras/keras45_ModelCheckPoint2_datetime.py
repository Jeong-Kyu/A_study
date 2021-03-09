# 인공지능계의 helli world - mnist

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)#(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)#(10000, 28, 28) (10000,)

print(x_train[0])
print('y_train[0] : ', y_train[0])
print(x_train[0].shape)#(28, 28)

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

from tensorflow.keras.models import Sequential
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

# ModelCheckPoint
import datetime
date_now = datetime.datetime.now()
# print(date_now)
# date_time = date_now.strftime("%m%d_%H%M")
# print(date_time)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
filepath = '../data/modelcheckpoint/'
filename = '_{epoch:02d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath,"k45_",'{timer}',filename])

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.distribute import distributed_file_utils
@keras_export('keras.callbacks.ModelCheckpoint')
class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        try:
        # `filepath` may contain placeholders such as `{epoch:02d}` and
        # `{mape:.2f}`. A mismatch between logged metrics and the path's
        # placeholders can cause formatting to fail.
            file_path = self.filepath.format(epoch=epoch + 1, timer=datetime.datetime.now().strftime('%m%d_%H%M'), **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". '
                        'Reason: {}'.format(self.filepath, e))
        self._write_filepath = distributed_file_utils.write_filepath(
            file_path, self.model.distribute_strategy)
        return self._write_filepath

# modelpath = '../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor='val loss', patience=5)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=128, epochs=50, validation_split=0.2, callbacks=[es, cp])

loss, acc = model.evaluate(x_test, y_test, batch_size=200)
print('loss : ', loss)
print('accuracy : ', acc)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic"
plt.figure(figsize=(10, 6))

plt.subplot(2,1,1)  # 2행 1열 중 첫번째
plt.plot(hist.history['loss'], marker= '.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker= '.', c='blue', label='val_loss')
plt.grid()

plt.title('손실Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') #upper right가 없을 경우 빈 공간을 찾아감

plt.subplot(2,1,2) # 2행 2열 중 두번쨰
plt.plot(hist.history['accuracy'], marker= '.', c='red')
plt.plot(hist.history['val_accuracy'], marker= '.', c='blue')
plt.grid()

plt.title('정확Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy','val_accuracy']) #legend에 직접 라벨명을 넣으면 표기

plt.show()