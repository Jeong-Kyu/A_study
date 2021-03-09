# ImageDataGenerator fit_generator

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D,Flatten
from sklearn.decomposition import PCA

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)
xy_train = train_datagen.flow_from_directory(
    # 첫번 째 인자는 디렉토리 폴더를 받음
    '../data2', # (160,256,256,3)
    # 폴더 안에 있는 전체 이미지 데이터를 사이즈를 바꿈
    target_size=(150,150), # (160,150,150,3)
    batch_size=5, 
    class_mode='binary'
)

xy_test = test_datagen.flow_from_directory(
    # 첫번 째 인자는 디렉토리 폴더를 받음
    '../data2', # (120,256,256,3)
    # 폴더 안에 있는 전체 이미지 데이터를 사이즈를 바꿈
    target_size=(150,150), # (120,150,150,3)
    batch_size=5, 
    class_mode='binary'
)

model = Sequential()
model.add(Conv2D(128,(3,3), input_shape=(150,150,3)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3)))
model.add(Dropout(0.2))
model.add(Conv2D(16,(3,3)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='loss', patience=10, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit_generator(
    xy_train, steps_per_epoch=32, epochs=200, validation_data=xy_test, validation_steps=4, callbacks=[es,reduce_lr]
)
# history = model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test), callbacks=[es,reduce_lr])

# fit_generator -> xy together
# step_per_epoch -> data / batch_size
loss, acc = model.evaluate(xy_test)
print("loss : ", loss)
print("acc : ", acc)

# loss :  0.6928289532661438
# acc :  0.5155529975891113

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# import matplotlib.pyplot as plt

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])

# plt.title('loss & acc')
# plt.ylabel('loss & acc')
# plt.xlabel('epoch')
# plt.legend(['train loss', 'val loss', 'train acc', 'val acc']) # 순서대로 설명
# plt.show()

