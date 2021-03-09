import numpy as np
import pandas as pd
'''
# 자료불러오기
df = pd.read_csv('./samsung/csv/삼성전자.csv',thousands = ',', encoding='cp949', index_col=0, header=0)
df2 = pd.read_csv('./samsung/csv/삼성전자2.csv',thousands = ',', encoding='cp949', index_col=0, header=0)
# 자료 분할
sj1 = df[1:662]

sj1 = sj1.drop(sj1.columns[[4,5,6,7,9,10,11,13]], axis='columns')
sj2 = df2[['시가','고가','저가','종가','개인','프로그램']]
sj2 = sj2[:2]
sj = sj2.append(sj1, ignore_index=True)
sj = sj.astype('float32')
sj = sj[::-1]


# 상관계수를 통한 자료 선정
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()

# print(sj)

# 데이터
sj=sj.to_numpy()
'''
sj = np.load('./samsung/save/data/sam_210115_xy.npy')

x=sj[:-1, :]
y=sj[1:, 3]

# print(x.shape) 
# print(y.shape) 

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


# 모델링
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, Dropout, LSTM
'''
model = Sequential()
model.add(LSTM(500, input_shape=(6,1), activation='relu'))
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
es = EarlyStopping(monitor='val loss', patience=5, mode='auto')
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, batch_size=128, epochs=10000, validation_data=(x_val, y_val), callbacks=[es, cp])
'''
model = load_model('./samsung/save/samsung_fitsave/sam_210115_fitsave.h5')

result = model.evaluate(x_test, y_test, batch_size=20)
print('loss : ', result)

x_predict = np.array([[88700,90000,88700,89700,3245719.0,-1091335.0]])
x_predict = scaler.transform(x_predict)
x_predict = x_predict.reshape(1,x_train.shape[1],1)
y_predict = model.predict(x_predict)
print(y_predict)
