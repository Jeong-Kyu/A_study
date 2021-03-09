import numpy as np
import pandas as pd

wine = pd.read_csv('C:\data\csv\winequality-white.csv',sep=';')

wine_np = wine.values
# x = wine_np[:,:-1]
# y = wine_np[:,-1]

y = wine['quality']
x = wine.drop('quality',axis=1)

print(y[100:101])
newlist = []
for i in list(y):
    if i <= 4:
        newlist += [0]
    elif i <= 7:
        newlist +=[1]
    else:
        newlist +=[2]
y = newlist

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

# score :  0.95



