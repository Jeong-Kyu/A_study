# 61 -> pipeline


# w save
# model.save()
# pickle

#lstm
#parameter change
#nod

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten, MaxPool2D, LSTM
from tensorflow.keras.datasets import mnist, boston_housing
from sklearn.datasets import load_diabetes, load_wine

dataset = load_wine()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split,cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )

# 1
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline

# 2
def build_model(drop = 0.5, optimizer = 'adam'):
    inputs = Input(shape=(13,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128,activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['accuracy'], loss='categorical_crossentropy')

    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"clf__batch_size":batches, "clf__optimizer":optimizers, "clf__drop":dropout}
hyperparameters = create_hyperparameters()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model2 = KerasClassifier(build_fn=build_model, epochs=100, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# pipe = make_pipeline(MinMaxScaler(), model2)
pipe = Pipeline([("scaler", MinMaxScaler()),("clf", model2)])
search = RandomizedSearchCV(pipe, hyperparameters, cv = 3)
search.fit(x_train, y_train)
score = search.score(x_test,y_test)
print(score)
print(search.best_params_)
print(search.best_estimator_)
print(search.best_score_)
