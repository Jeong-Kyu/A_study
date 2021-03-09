import numpy as np

x_train = np.load('../data/npy/fashion_mnist_x_train.npy')
x_test = np.load('../data/npy/fashion_mnist_x_test.npy')
y_train = np.load('../data/npy/fashion_mnist_y_train.npy')
y_test = np.load('../data/npy/fashion_mnist_y_test.npy')

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=66, shuffle=True)

x_train = x_train.reshape(48000,28*28)/225.
x_test = x_test.reshape(10000,28*28)/225.
x_val = x_val.reshape(12000,28*28)/225.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,Flatten

#2. 모델링
model = Sequential()
model.add(Dense(30,input_shape=(28*28,), activation='relu'))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(1))
#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics = ['mae'])
'''
EarlyStopping
'''
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k46_1_fashion_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto') #loss값이 가장낮을 때를 10번 지나갈 때 까지 기다렸다가 stop. mode는 min, max, auto조정가능
hist = model.fit(x_train, y_train, epochs=10, batch_size=1, validation_data= (x_val, y_val), callbacks = [early_stopping,cp])


#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test) 
#print(y_predict)


# cnn
# loss :  1.6982073783874512
# mae :  0.9694334864616394
# RMSE : 1.3031527
# R2 :  0.7941567117430608

# dnn
# loss :  1.5654747486114502
# mae :  0.8545660376548767
# RMSE : 1.2511892
# R2 :  0.8102455358011839