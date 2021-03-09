import numpy as np

x = np.array([range(100), range(301, 401), range(1, 101), range(111, 211), range(11, 111)])
y = np.array([range(711, 811), range(1,101)])

x = np.transpose(x)
y = np.transpose(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=32)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# input1=Input(shape=(5,))
# a1=Dense(3, activation='relu')(input1)
# a2=Dense(4)(a1)
# output1=Dense(2)(a2)
# model=Model(inputs=input1, outputs=output1)
# model.summary()

model=Sequential()
model.add(Dense(3, activation='relu', input_shape=(5,)))
model.add(Dense(4))
model.add(Dense(2))


model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.2)

loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import  mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test,y_predict))
print("MSE : ", mean_squared_error(y_test,y_predict))

from sklearn.metrics import r2_score
print("R2 : ", r2_score(y_test, y_predict))

