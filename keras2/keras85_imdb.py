from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train),(x_test, y_test) = imdb.load_data(
    num_words=2000
)
# print('==========================')
# print(x_train.shape, y_train.shape) #(25000,) (25000,)
# print(x_test.shape, y_test.shape) #(25000,) (25000,)
# print(x_train[0])
# print(y_train[0])
#임베딩 모델 만들기!

# print('최대길이 : ', max(len(l) for l in x_train)) # 2494
# print('평균길이 : ', sum(map(len, x_train))/len(x_train)) # 238.71364
# plt.hist([len(s) for s in x_train], bins = 50)
# plt.show() ------1000

# y분포
# unique_elemnets, counts_elements = np.unique(y_train, return_counts=True)
# print('y분포 : ', dict(zip(unique_elemnets, counts_elements)))
# print('================================================')
# plt.hist(y_train, bins = 46)
# plt.show() #---------- 0,1 이진분류

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=300) #post
x_test = pad_sequences(x_test, padding='pre', maxlen=300) #post
print(x_train.shape)
print(x_test.shape)
# x_train = x_train.reshape()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D, Dropout, Conv2D, BatchNormalization, MaxPooling1D # 원핫은 너무 커져서 임베딩을 사용

#임베딩 레이어 제거

model = Sequential()
model.add(Embedding(input_dim=2000, output_dim=128, input_length=300))
model.add(LSTM(32))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=100,  validation_split = 0.2, callbacks=[early_stopping])

acc =model.evaluate(x_test, y_test)[1]
print(acc)

# LSTM
# 0.8551599979400635
# Conv1D
# 0.8539199829101562
# (LSTM)Nadam
# 0.8608400225639343
# (LSTM)rmsprop
# 0.8682000041007996
# (LSTM+Dense)rmsprop
# 0.8688399791717529
# (LSTM+Dense)rmsprop + input_dim 2000
# 0.8690000176429749
# (LSTM+Dense)rmsprop + input_dim 2000 + input_length 300
# 0.8707600235939026
