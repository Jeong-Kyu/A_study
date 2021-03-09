from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = reuters.load_data(
    num_words=5000, test_split=0.2
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=500) #post
x_test = pad_sequences(x_test, padding='pre', maxlen=500) #post

print(x_train.shape) # (13, 5)
# pad_x = pad_x.reshape(13,5,1)
print(np.unique(x_train))
print(len(np.unique(x_train))) # 28 / 0~27에서 11 제외 (maxlen =4 때문에)

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# encoder = OneHotEncoder()
# encoder.fit(y_train)
# y_train = encoder.transform(y_train)
# y_test = encoder.transform(y_test)
# y_train=y_train.toarray()
# y_test=y_test.toarray()
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)        # y도 원핫인코딩 꼭 하기
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D, Dropout # 원핫은 너무 커져서 임베딩을 사용

#임베딩 레이어 제거

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=500)) 
# model.add(Embedding(28,11)) # None, None, 11 임으로 안먹힘
# model.add(LSTM(128, activation='tanh'))
model.add(Conv1D(filters=32, kernel_size=2, activation='tanh'))
model.add(Conv1D(filters=32, kernel_size=2, activation='tanh'))
model.add(Flatten())
model.add(Dense(32, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(46, activation='softmax'))
model.summary()


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train,batch_size = 32, validation_split=0.2 ,epochs=100, callbacks=[early_stopping])

acc =model.evaluate(x_test, y_test)[1]
print(acc)

# LSTM
# 0.6807658076286316
# Conv1D
# 0.6473730802536011