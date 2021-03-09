from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

model = VGG16(weights = 'imagenet', include_top=False, input_shape=(32,32,3)) #원하는 사이즈는 include_top=False / 디폴트 224*224
# print(model.weights)

model.trainable = False
model.summary()
print(len(model.weights))           # 26
print(len(model.trainable_weights)) # 0
'''
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
'''
model.trainable = True
model.summary()
print(len(model.weights))           # 26
print(len(model.trainable_weights)) # 26
'''
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
'''
