import numpy as np

x_data = np.load('../data/npy/diabetes_x.npy')
y_data = np.load('../data/npy/diabetes_y.npy')

print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)

#1_2. 데이터 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8, shuffle = True, random_state = 66 )
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size= 0.8, shuffle = True, random_state = 66, )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
#2. 모델링
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
model = Sequential()
model.add(Dense(100, input_dim=10, activation = 'relu')) # 기본값 : activation='linear' 
model.add(Dropout(0.4))
model.add(Dense(75, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
'''
EarlyStopping
'''
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k46_5_diabets_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=7, validation_data= (x_val, y_val), callbacks = [early_stopping, cp])

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)



#데이터 전처리 전
# loss :  3317.64599609375
# mae :  47.06387710571289
# RMSE :  57.59901189718801
# R2 :  0.488809627121195

#데이터 엉망 처리 후
# loss :  3379.458984375
# mae :  47.35618591308594
# RMSE :  58.13311275393621
# R2 :  0.47928539874511966

#데이터 x를 전처리한 후
# loss :  3291.452880859375
# mae :  46.496116638183594
# RMSE :  57.37118551454562
# R2 :  0.49284554101046385

#데이터 x_train잡아서 전처리한 후....
# loss :  3421.5537109375
# mae :  47.82155227661133
# RMSE :  58.49405010020266
# R2 :  0.47279929140593635

#발리데이션 test분리
# loss :  3369.262451171875
# mae :  48.33604431152344
# RMSE :  58.04534944194592
# R2 :  0.5128401315682825

#Earlystopping 적용
# loss :  57.708213806152344
# mae :  5.144794464111328
# RMSE :  7.596591897809446
# R2 :  0.991656001135139

# model = Sequential()
# model.add(Dense(100, input_dim=10, activation = 'relu')) # 기본값 : activation='linear' 
# model.add(Dense(75, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(1))
# loss :  14.287396430969238
# mae :  3.0699069499969482
# RMSE :  3.779867213295329
# R2 :  0.9979341930647975

# loss :  6162.50048828125
# mae :  62.28391647338867
# RMSE :  78.50159664503691
# R2 :  0.05046805612688887