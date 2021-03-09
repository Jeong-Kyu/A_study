# ImageDataGenerator fit
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D,Flatten, BatchNormalization
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
xy_all = test_datagen.flow_from_directory(
    '../data2',
    target_size=(150,150),
    batch_size=2000,
    class_mode='binary'
)
print(xy_all)
np.save('../data2/keras67_x.npy', arr=xy_all[0][0])
np.save('../data2/keras67_y.npy', arr=xy_all[0][1])
print("save") #(1388, 150, 150, 3)

x = np.load('../data2/keras67_x.npy')
y = np.load('../data2/keras67_y.npy')
print("load") #(1388, 150, 150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

print(x.shape) #(1736, 150, 150, 3)
print(y.shape) #(1736,)
print(x_train.shape) #(1388, 150, 150, 3)
print(x_test.shape) #(348, 150, 150, 3)


model = Sequential()
model.add(Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3),padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3)))
model.add(Dropout(0.3))

model.add(Flatten()) #2차원
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='loss', patience=10, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=200, validation_data=(x_test, y_test), callbacks=[es,reduce_lr])

loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

# loss :  0.6835646629333496
# acc :  0.7701149582862854
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

# loss :  0.6835646629333496
# acc :  0.7701149582862854