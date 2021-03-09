#lstm
#parameter change
#nod

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPool2D, LSTM
from tensorflow.keras.datasets import mnist, boston_housing
from sklearn.datasets import load_diabetes, load_breast_cancer

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

# 1
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)
print(x_test.shape)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train).reshape(455, 30) #x_train만 trans 후 바뀐수치 x_train에 다시넣기
x_test = scaler.transform(x_test).reshape(114, 30) #x_test 따로 trans  후 바뀐수치 x_test에 다시넣기
 

# 2
def build_model(drop = 0.5, optimizer = 'adam'):
    inputs = Input(shape=(30,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(2, activation='sigmoid', name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['accuracy'], loss='binary_crossentropy')

    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"batch_size":batches, "optimizer":optimizers, "drop":dropout}
hyperparameters = create_hyperparameters()
# model2 = build_model()

nod = 10
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv = 3)
# search = GridSearchCV(model2, hyperparameters, cv = 3)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = '../data/modelcheckpoint/k62_cancer_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

search.fit(x_train, y_train, verbose=1, epochs=100, validation_split=0.2, callbacks=[es, reduce_lr, cp])

print(search.best_params_) # ㄴㅐㄱㅏ ㅅㅓㄴㅌㅐㄱㅎㅏㄴ ㅍㅏㄹㅏㅁㅣㅌㅓ ㅈㅜㅇㅇㅔㅅㅓ
print(search.best_estimator_) # ㅈㅓㄴㅊㅔ ㅍㅏㄹㅏㅁㅣㅌㅓ ㅈㅜㅇㅇㅔㅅㅓ
print(search.best_score_)
acc = search.score(x_test, y_test)
print("final score : ", acc)

# {'optimizer': 'adam', 'drop': 0.1, 'batch_size': 50}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000029B057484C0>
# 0.9648541808128357
# 3/3 [==============================] - 0s 1ms/step - loss: 0.1754 - accuracy: 0.9561
# final score :  0.9561403393745422