import numpy as np
import pandas as pd

wine = pd.read_csv('C:\data\csv\winequality-white.csv',sep=';')
# print(wine)
# print(wine['quality'].value_counts())

wine_np = wine.values
x = wine_np[:,:-1]
y = wine_np[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

model = RandomForestClassifier()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('score : ',score)

