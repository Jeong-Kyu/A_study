# keras67_1 남자 여자에 를 넣어서 잡음 제거

# ImageDataGenerator fit
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D,Flatten, BatchNormalization, Conv2DTranspose, LeakyReLU, Input
from sklearn.decomposition import PCA

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=5,
#     zoom_range=1.2,
#     shear_range=0.7,
#     fill_mode='nearest'
# )

# test_datagen = ImageDataGenerator(rescale=1./255)
# xy_all = test_datagen.flow_from_directory(
#     '../data2',
#     target_size=(150,150),
#     batch_size=2000,
#     class_mode='binary'
# )
# print(xy_all)
# np.save('../data2/keras67_x.npy', arr=xy_all[0][0])
# np.save('../data2/keras67_y.npy', arr=xy_all[0][1])
# print("save") #(1388, 150, 150, 3)

x = np.load('../data2/keras67_x.npy')
y = np.load('../data2/keras67_y.npy')
print("load") #(1388, 224, 224, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

x_train_noised = x_train + np.random.normal(0,0.1,size = x_train.shape)
x_test_noised = x_test + np.random.normal(0,0.1,size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

print(x_test.shape)
print(x_test_noised.shape)


def autoencoder():
    model = Sequential()
    model.add(Conv2D(256, 3, activation= 'relu', padding= 'same', input_shape = (224,224,3)))
    model.add(Conv2D(128, 5, activation= 'relu', padding= 'same'))
    model.add(Conv2D(64, 5, activation= 'relu', padding= 'same'))
    model.add(Conv2D(32, 5, activation= 'relu', padding= 'same'))
    model.add(Conv2D(3, 3, padding = 'same', activation= 'sigmoid'))

    return model

model = autoencoder() # 95% pca수치
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])
model.fit(x_train_noised,x_train,epochs=30,batch_size=4)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3,5,figsize=(20,7))

random_images = random.sample(range(output.shape[0]),5)

# 원본
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(224,224,3),cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(224,224,3),cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(224,224,3),cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


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