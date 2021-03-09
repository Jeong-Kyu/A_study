import numpy as np

x= np.array([range(100),range(100,200)])
y= np.array([range(200,300),range(300,400)])

x=np.transpose(x)
y=np.transpose(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 57)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(2, activation='relu', input_shape=(2,)))
model.add(Dense(2))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.2)

loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)
y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE : ', RMSE(y_test, y_pred))
print('mse : ', mean_squared_error(y_test, y_pred))
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print('R2 : ', r2)

print(y_pred)