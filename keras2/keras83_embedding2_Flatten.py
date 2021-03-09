from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밌어요", "참 최고예요", "참 잘 만든 영화예요", "추천하고 싶은 영화입니다", "한 번 더 보고 싶네요", "글쌔요",
"별로예요", "생각보다 지루해요", "연기가 어색해요", "재미없어요", "너무 재미없다", "참 재밋네요", "규현이가 잘 생기긴 했어요"]

#긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x) # [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) #post
print(pad_x.shape) # (13, 5)

print(np.unique(pad_x))
print(len(np.unique(pad_x))) # 28 / 0~27에서 11 제외 (maxlen =4 때문에)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D # 원핫은 너무 커져서 임베딩을 사용

model = Sequential()
model.add(Embedding(input_dim=28, output_dim=11, input_length=5)) 
# model.add(Embedding(28,11)) # None, None, 11 임으로 안먹힘
# model.add(LSTM(32))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels, epochs=100)

acc =model.evaluate(pad_x, labels)[1]
print(acc)