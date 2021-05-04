import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

x_train = x_train.reshape(404, 13).astype('float32')/255.
x_test = x_test.reshape(102, 13).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=2,
    loss = 'mse',
    metrics = ['mae'],

)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', mode='min', patience=6)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor = 0.5, verbose=2)
ck = ModelCheckpoint('C:/data/modelcheckpoint', save_weights_only=True, save_best_onlT=True, monitor='val_loss', verbose=1) 
model.fit(x_train, y_train, epochs=3, validation_split=0.2,
        callbacks = [es,lr,ck])

results = model.evaluate(x_test, y_test)

print(results)


# model.summary()