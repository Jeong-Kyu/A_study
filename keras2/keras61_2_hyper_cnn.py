#cnn
#parameter change
#nod

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255.

# 2
def build_model(drop = 0.5, optimizer = 'adam'):
    inputs = Input(shape=(28,28,1), name='input')
    x = Conv2D(filters=512, kernel_size=(2,2) ,activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(filters=256, kernel_size=(2,2), activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Conv2D(filters=128, kernel_size=(2,2), activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    x = MaxPool2D(pool_size=(3,3))(x)
    x = Flatten()(x)
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
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000015DC2FDDE50>
# 0.9785166581471761
#   1/200 [..............................] - ETA: 1s - loss: 0.0356 - acc: 0.9800WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.0010s vs `on_test_batch_end` time: 0.0060s). Check your callbacks.
# 200/200 [==============================] - 2s 8ms/step - loss: 0.0531 - acc: 0.9846
# final score :  0.9846000075340271