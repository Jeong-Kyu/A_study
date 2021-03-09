import numpy as np

a = np.array(range(1,11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append(subset)    #[item for item in subset]
    return np.array(aaa)

dataset = split_x(a,size)

x = dataset[:,:4]
y = dataset[:,-1]

print(x) #(6,4)
print(y) #(6,1)

x = x.reshape(6,4,1)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input,LSTM

input1 = Input(shape =(4,1))
dense1 = LSTM(10, activation='relu')(input1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(1, activation='relu')(dense1)

model = Model(inputs = input1, outputs= dense1)

model.compile(loss = 'mse', optimizer= 'adam', metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=10)


loss = model.evaluate(x,y, batch_size=1)

print("model.metrics_name : ", model.metrics_names)
print(loss)

x_pred = np.array([7,8,9,10])
x_pred = x_pred.reshape(1,4,1)
y_pred = model.predict(x_pred)
print(y_pred)

# LSTM
# model.metrics_name :  ['loss', 'mae']
# [0.020387491211295128, 0.10782583802938461]
# [[11.410588]]