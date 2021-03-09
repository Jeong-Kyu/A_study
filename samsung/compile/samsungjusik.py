import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

# 자료불러오기
df = pd.read_csv('./samsung/csv/삼성전자.csv',thousands = ',', encoding='cp949', index_col=0, header=0)
df2 = pd.read_csv('./samsung/csv/삼성전자2.csv',thousands = ',', encoding='cp949', index_col=0, header=0)
# 자료 분할
sj1 = df[1:662]
sj1 = sj1.drop(sj1.columns[[5,6,7,8,10,11,13]], axis='columns')
sj2 = df2[['시가','고가','저가','종가','등락률','기관','프로그램']]
sj2 = sj2[:2]
sj = sj2.append(sj1, ignore_index=True)
sj = sj.astype('float32')
sj.to_csv('./samsung/csv/0114삼성전자.csv', encoding='utf-8-sig')
sj = sj[::-1]
'''
# 상관계수를 통한 자료 선정
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()

# print(sj)
'''
# 데이터
sj=sj.to_numpy()
np.save('./samsung/save/data/sam_210100_xy.npy', arr=sj)

x=sj[:-1, :]
y=sj[1:, 3]

print(type(x)) 
print(y.shape) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=66, shuffle=True)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

# print(x_train)
# print(x_train.shape)
# print(x_test.shape)

print(type(x_train)) 

# 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, Dropout, LSTM

model = Sequential()
model.add(LSTM(500, input_shape=(7,1), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './samsung/save/modelcheckpoint/sam15_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor='val_loss', patience=500, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train, y_train, batch_size=128, epochs=30000, validation_data=(x_val, y_val), callbacks=[es, cp])

model.save('./samsung/save/samsung_fitsave/sam_210100_fitsave.h5')

result = model.evaluate(x_test, y_test, batch_size=20)
print('loss : ', result)

x_predict = np.array([[88700,90000,88700,89700,-0.99,3245719.0,-1091335.0]])
x_predict = scaler.transform(x_predict)
x_predict = x_predict.reshape(1,x_train.shape[1],1)
y_predict = model.predict(x_predict)
print(y_predict)

# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['axes.unicode_minus'] = False 
# matplotlib.rcParams['font.family'] = "Malgun Gothic"
# plt.figure(figsize=(10, 6))

# plt.subplot(2,1,1)  # 2행 1열 중 첫번째
# plt.plot(hist.history['loss'], marker= '.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker= '.', c='blue', label='val_loss')
# plt.grid()

# plt.show()

# loss :  3042318.0
# [[90635.73]]
