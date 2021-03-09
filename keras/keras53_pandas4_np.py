import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0)

print(df)

print(df.shape) #(150, 5)
print(df.info())

aaa=df.to_numpy()
print(aaa)
print(type(aaa))

bbb=df.values
print(bbb)
print(type(bbb))

np.save('../data/npy/iris_sklearn.npy', arr=aaa)

#과제
#판다스의 loc, iloc에 대해 정리