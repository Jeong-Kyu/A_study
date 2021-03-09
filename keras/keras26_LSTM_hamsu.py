# keras23_LStM3_scale 함수형으로

import numpy as np
# 1datetime A combination of a date and a time. Attributes: ()
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

print(x.shape) #(13,3)
print(y.shape) #(13, )

x = x.reshape(13,3,1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input


input1 = Input(shape = (3,1))
a1 = LSTM(10, activation='relu')(input1)
a2 = Dense(500)(a1)
a3 = Dense(500)(a2)
a4 = Dense(50)(a3)
a5 = Dense(1000)(a4)
output1 = Dense(1)(a5)
model = Model(inputs = input1, outputs = output1)


model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, epochs=500, batch_size=1)

loss=model.evaluate(x_test,y_test)
print(loss)

x_pred=x_pred.reshape(1,3,1)
y_pred=model.predict(x_pred)
print(y_pred)

# Sequential
# 0.07824796438217163
# [[80.906136]]

#함수형
# 0.0325588695704937
# [[79.10631]]