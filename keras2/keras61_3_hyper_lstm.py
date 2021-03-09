#lstm
#parameter change
#nod

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPool2D, LSTM
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28,28).astype('float32')/255.
x_test = x_test.reshape(10000, 28,28).astype('float32')/255.

# 2
def build_model(drop = 0.5, optimizer = 'adam'):
    inputs = Input(shape=(28,28), name='input')
    x = LSTM(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')

    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"batch_size":batches, "optimizer":optimizers, "drop":dropout}
hyperparameters = create_hyperparameters()
# model2 = build_model()

nod = 10
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv = 3)
# search = GridSearchCV(model2, hyperparameters, cv = 3)


search.fit(x_train, y_train, verbose=1)

print(search.best_params_) # ㄴㅐㄱㅏ ㅅㅓㄴㅌㅐㄱㅎㅏㄴ ㅍㅏㄹㅏㅁㅣㅌㅓ ㅈㅜㅇㅇㅔㅅㅓ
print(search.best_estimator_) # ㅈㅓㄴㅊㅔ ㅍㅏㄹㅏㅁㅣㅌㅓ ㅈㅜㅇㅇㅔㅅㅓ
print(search.best_score_)
acc = search.score(x_test, y_test)
print("final score : ", acc)

# {'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 40}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001932322BFA0>
# 0.958816667397817
# 250/250 [==============================] - 0s 910us/step - loss: 0.1148 - acc: 0.9696
# final score :  0.9696000218391418

# {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 50}
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001E2BE18B5E0>
# 0.9417666594187418
# 200/200 [==============================] - 1s 5ms/step - loss: 0.1329 - acc: 0.9575
# final score :  0.9574999809265137