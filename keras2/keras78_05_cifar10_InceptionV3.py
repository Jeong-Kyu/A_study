#실습
# cifar10으로 vgg16만들기
#결과치 비교
import numpy as np
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)# (50000,32,32,3)
print(x_test.shape) # (50000,1)

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

inceptionV3 = InceptionV3(weights = 'imagenet', include_top=False, input_shape=(32,32,3)) #원하는 사이즈는 include_top=False / 디폴트 224*224
# print(model.weights)

inceptionV3.trainable = False
# vgg16.summary()
# print(len(vgg16.weights))           # 26
# print(len(vgg16.trainable_weights)) # 0

model = Sequential()
model.add(inceptionV3)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))#, activation='softmax'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics = ['mae'])
'''
EarlyStopping
'''
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto') #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
model.fit(x_train, y_train, epochs=10, batch_size=1, validation_data= (x_test, y_test), callbacks = [early_stopping])

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae)

# loss :  8.250359535217285
# mae :  2.4999992847442627

# loss :  28.49924659729004
# mae :  4.499936103820801

# loss :  6.089522838592529
# mae :  1.9320931434631348

# loss :  6.196646690368652
# mae :  2.0252695083618164

# loss :  7.561173439025879
# mae :  2.3440632820129395