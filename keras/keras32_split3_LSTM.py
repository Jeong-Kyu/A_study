#LSTM


# data 1~100
#       x          y
#  1,2,3,4,5       6
#.....
#95,96,97,98,99  100

#predict
#96,97,98,99,100 --> 101
#......
#100,101,102,103,104 --> 105
# ex predict(101,102,103,104,105)

import numpy as np

a = np.array(range(1,101))
size = 6

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append(subset)    #[item for item in subset]
    return np.array(aaa)

dataset = split_x(a,size)
b = np.array(range(96,105))
x_predict = split_x(b,5)


x= dataset[:,:5]
y= dataset[:,-1]

print(x.shape)
print(y.shape)
print(x_predict.shape)




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=33)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=33)




from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) #x_train만 trans 후 바뀐수치 x_train에 다시넣기
x_test = scaler.transform(x_test) #x_test 따로 trans  후 바뀐수치 x_test에 다시넣기
x_val = scaler.transform(x_val)
x_predict = scaler.transform(x_predict)

# print(x_train.shape)
# print(x_test.shape)
# print(x_val.shape)


x_train = x_train.reshape(60,5,1)
x_test = x_test.reshape(19,5,1)
x_val = x_val.reshape(16,5,1)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input,LSTM

input1 = Input(shape =(5,1))
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

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')


model.fit(x_train, y_train, epochs=500, batch_size=10,validation_data=(x_val,y_val), callbacks = [early_stopping])


loss = model.evaluate(x_test,y_test, batch_size=1)

print("model.metrics_name : ", model.metrics_names)
print(loss)


x_predict = x_predict.reshape(5,5,1)
y_pred = model.predict(x_predict)
print(y_pred)

# LSTM
# model.metrics_name :  ['loss', 'mae']
# [0.005712755024433136, 0.06309276074171066]
# [[100.93703 ]
#  [101.921906]
#  [102.90616 ]
#  [103.889786]
#  [104.8728  ]]

# # test/val/early_stopping
# model.metrics_name :  ['loss', 'mae']
# [132.39675903320312, 10.755218505859375]
# [[82.00972 ]
#  [81.921425]
#  [81.72536 ]
#  [81.39647 ]
#  [80.91113 ]]

# epochs up 100->500
# model.metrics_name :  ['loss', 'mae']
# [0.5017098784446716, 0.699554443359375]
# [[100.093544]
#  [101.07139 ]
#  [102.0027  ]
#  [102.93279 ]
#  [103.863075]]

# MinMaxScaler
# model.metrics_name :  ['loss', 'mae']
# [0.09059702605009079, 0.17118310928344727]
# [[102.14428 ]
#  [103.314964]
#  [104.49621 ]
#  [105.688   ]
#  [106.89045 ]]