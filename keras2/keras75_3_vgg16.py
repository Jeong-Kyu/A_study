from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

vgg16 = VGG16(weights = 'imagenet', include_top=False, input_shape=(32,32,3)) #원하는 사이즈는 include_top=False / 디폴트 224*224
# print(model.weights)

vgg16.trainable = False
vgg16.summary()
print(len(vgg16.weights))           # 26
print(len(vgg16.trainable_weights)) # 0

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))#, activation='softmax'))
model.summary()

print('그냥 가중치의 수 : ', len(model.weights))           # 32
print('동결하기 전 훈련되는 가중치의 수 : ', len(model.trainable_weights)) # 6

####################요기 하단때문에

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns= [ 'Layer Type', 'Layer Name', 'Layer Trainable'])

print(aaa)