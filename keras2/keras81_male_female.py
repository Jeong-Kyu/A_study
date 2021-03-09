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
    target_size=(224,224),
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
from tensorflow.keras.applications import VGG19, VGG16, InceptionV3
inceptionV3 = VGG16(weights = 'imagenet', include_top=False, input_shape=(224,224,3)) #원하는 사이즈는 include_top=False / 디폴트 224*224


model = Sequential()
model.add(inceptionV3)
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

from tensorflow.keras.preprocessing.image import load_img, img_to_array
# 이미지 불러오기
img = load_img('C:\data\image\pred_00.png', target_size=(224,224))

import matplotlib.pyplot as plt
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
from tensorflow.keras.applications.vgg16 import preprocess_input
x = preprocess_input(x)
images = np.vstack([x])
classes = model.predict(images, batch_size=2)
print(classes)
if classes>0.5:
    print("그는",int(100*acc),"% 확률로 남자다")
else:
    print( "그녀는",int(100*acc),"% 확률로 여자다")
# Conv2D
# loss :  0.6835646629333496
# acc :  0.7701149582862854
# VGG16
# loss :  0.6769675016403198
# acc :  0.8045976758003235
# VGG19
# loss :  0.830842912197113
# acc :  0.7241379022598267
# InceptionV3
# loss :  0.6854262948036194
# acc :  0.7758620977401733