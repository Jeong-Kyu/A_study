from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

print(type(x_train[0])) #<class 'list'>
print('==========================')
print(x_train.shape, y_train.shape) #(8982,) (8982,)
print(x_test.shape, y_test.shape) #(2246,) (2246,)
print('뉴스기사 최대길이 : ', max(len(l) for l in x_train)) # 뉴스기사 최대길이 :  2376
print('뉴스기사 평균길이 : ', sum(map(len, x_train))/len(x_train)) # 뉴스기사 평균길이 :  145.5398574927633
plt.hist([len(s) for s in x_train], bins = 50)
plt.show()

# y분포
unique_elemnets, counts_elements = np.unique(y_train, return_counts=True)
print('y분포 : ', dict(zip(unique_elemnets, counts_elements)))
print('================================================')
# plt.hist(y_train, bins = 46)
# plt.show()


# x의 단어 분포
word_to_index = reuters.get_word_index()
print(word_to_index)
print(type(word_to_index))
print('================================================')

# 키와 밸류 교체
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key
# 키 밸류 교환후
print(index_to_word[1]) # the
print(index_to_word[30979]) # northerly
print(len(index_to_word)) # 30979

# x_train[0] 복원
print(x_train[0])
print(' '.join([index_to_word[index] for index in x_train[0]]))

# y카테고리 갯수 출력
category = np.max(y_train)+1
print('y 카테고리 갯수: ', category) # 46

# y의 유니크한 값 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)

########################전처리###########################
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
print(x_train.shape, x_test.shape) #(8982, 100) (2246, 100)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape)

#모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,LSTM,Conv1D,Flatten

model = Sequential()
model.add(Embedding(10000,64))
model.add(LSTM(32))
model.add(Dense(46, activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size = 32, verbose=1)

results = model.evaluate(x_test, y_test)