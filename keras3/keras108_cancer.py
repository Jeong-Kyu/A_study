import numpy as np
import tensorflow as tf
import autokeras as ak
from sklearn.datasets import load_breast_cancer

# 1. 데이터
datesets = load_breast_cancer()
x = datesets.data
y = datesets.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

x_train = x_train.reshape(455,30).astype('float32')/255.
x_test = x_test.reshape(114,30).astype('float32')/255.

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

model = ak.StructuredDataClassifier(
    overwrite=True,
    max_trials=3,
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', mode='min', patience=6)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor = 0.5, verbose=2)
ck = ModelCheckpoint('C:/data/modelcheckpoint', save_weights_only=True, save_best_onlT=True, monitor='val_loss', verbose=1) 
model.fit(x_train, y_train, epochs=100, validation_split=0.2,
        callbacks = [es,lr,ck])

results = model.evaluate(x_test, y_test)

print(results)


# model.summary()