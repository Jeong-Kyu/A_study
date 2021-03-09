import numpy as np

# 1
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2
model = Sequential()
model.add(Dense(1000, input_dim =1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# optimizer = Adam(lr=0.01)
# loss :  2.33058011908302e-13 pred :  [[11.]]
# optimizer = Adam(lr=0.001)
# loss :  2.2761882974009495e-06 pred :  [[11.002592]]
# optimizer = Adam(lr=0.0001)
# loss :  5.678911293216515e-06 pred :  [[10.999139]]

# optimizer = Adadelta(lr=0.01)
# loss :  2.8951510103070177e-05 pred :  [[10.98841]]
# optimizer = Adadelta(lr=0.001)
# loss :  7.592088222503662 pred :  [[6.0535207]]
# optimizer = Adadelta(lr=0.0001)
# 31.066930770874023 pred :  [[1.1115918]]

# optimizer = Adamax(lr=0.01)
# loss :  5.8065553233677125e-12 pred :  [[11.000003]]
# optimizer = Adamax(lr=0.001)
# loss :  2.715978483780468e-11 pred :  [[10.99999]]
# optimizer = Adamax(lr=0.0001)
# loss :  0.0033418205566704273 pred :  [[10.921839]]

# optimizer = Adagrad(lr=0.01)
# loss :  2.546465111663565e-06 pred :  [[11.001892]]
# optimizer = Adagrad(lr=0.001)
# loss :  1.7594107703189366e-05 pred :  [[10.9970045]]
# optimizer = Adagrad(lr=0.0001)
# loss :  0.003955760970711708 pred :  [[10.921936]]

# optimizer = RMSprop(lr=0.01)
# loss :  19.043380737304688 pred :  [[2.2814364]]
# optimizer = RMSprop(lr=0.001)
# loss :  0.00027763767866417766 pred :  [[11.020827]]
# optimizer = RMSprop(lr=0.0001)
# loss :  0.0022530951537191868 pred :  [[10.9336405]]




model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

#4
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])
print('loss : ', loss, 'pred : ', y_pred)