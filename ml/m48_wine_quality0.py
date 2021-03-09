# 실습
import numpy as np
import pandas as pd


wine = pd.read_csv('C:\data\csv\winequality-white.csv',sep=';')
# winequality = winequality[winequality.quality != 9]
# winequality = winequality[winequality.quality != 3]

# print(wine)
# print(wine['quality'].value_counts())

# 9       5
# 8     175
# 7     880
# 6    2198
# 5    1457
# 4     163
# 3      20

wine_np = wine.values


x = wine_np[:,:-1]
y = wine_np[:,-1]
# print(y)
# print(y.shape)

newlist = []
for i in list(y):
    if i <= 4:
        newlist += [0]
    elif i <= 7:
        newlist +=[1]
    else:
        newlist +=[2]
y = newlist
y= np.array(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=66, shuffle=True)


from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1])
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 2차원 데이터로 변환하기
y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
y_val = y_val.reshape(-1,1)

# 원-핫 인코딩 적용
encoder = OneHotEncoder()
encoder.fit(y_test)
y_test = encoder.transform(y_test)
encoder.fit(y_train)
y_train = encoder.transform(y_train)
encoder.fit(y_val)
y_val = encoder.transform(y_val)

y_test=y_test.toarray()
y_train=y_train.toarray()
y_val=y_val.toarray()
print(y_train[1])
print(x_train.shape) #(3134, 11, 1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, Dropout, LSTM, MaxPooling1D, Input
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline

model=Sequential()
model.add(Dense(1024, input_shape=(11,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
modelpath = 'C:\data\modelcheckpoint\wq.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor='val_acc', patience=50, mode='auto')
rl = ReduceLROnPlateau(monitor='val_acc',factor=0.3,patience=20)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, batch_size=128, epochs=30000, validation_data=(x_val, y_val), callbacks=[es, cp, rl])

result = model.evaluate(x_test, y_test, batch_size=20)
print('loss : ', result)

# # 2
# def build_model(drop = 0.5, optimizer = 'adam'):
#     inputs = Input(shape=(11,), name='input')
#     x = Dense(512, activation='relu', name='hidden1')(inputs)
#     x = Dropout(drop)(x)
#     x = Dense(256, activation='relu', name='hidden2')(x)
#     x = Dropout(drop)(x)
#     x = Dense(128,activation='relu', name='hidden3')(x)
#     x = Dropout(drop)(x)
#     outputs = Dense(7, activation='softmax', name='output')(x)

#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer=optimizer, metrics=['accuracy'], loss='categorical_crossentropy')

#     return model

# def create_hyperparameters():
#     batches = [10, 20, 30, 40, 50]
#     optimizers = ['rmsprop', 'adam', 'adadelta']
#     dropout = [0.1, 0.2, 0.3]
#     return {"clf__batch_size":batches, "clf__optimizer":optimizers, "clf__drop":dropout}
# hyperparameters = create_hyperparameters()

# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
# model2 = KerasClassifier(build_fn=build_model, epochs=100, verbose=1)

# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# # pipe = make_pipeline(MinMaxScaler(), model2)
# pipe = Pipeline([("scaler", MinMaxScaler()),("clf", model2)])
# search = RandomizedSearchCV(pipe, hyperparameters, cv = 3)
# search.fit(x_train, y_train)
# score = search.score(x_test,y_test)
# print(score)
# print(search.best_params_)
# print(search.best_estimator_)
# print(search.best_score_)









