import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()
# print(dataset.keys())
# print(dataset.values())
x = dataset.data
y = dataset.target
print(x)
print(y)

# print(x.shape, y.shape) # (150,4) (150, )
# print(type(x), type(y)) # <class 'numpy.ndarray'>


# df = pd.DataFrame(x, columns=dataset.feature_names)
df = pd.DataFrame(x, columns=dataset['feature_names'])
print(df)
print(df.shape)
print(df.columns)
print(df.index)
print(df.head()) # df[:5]
print(df.tail()) # df[-5:]
print(df.info())
print(df.describe())

df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print(df.columns)

# y컬럼 추가
print(df['sepal_length'])
df['Target'] = dataset.target
print(df) # (150,5)
# print(df.columns)
# print(df.index)
# print(df.tail())
# print(df.isnull().sum())
# print(df.describe())
# print(df['Target'].value_counts())

# 상관계수 히트맵
print(df.corr())

# import matplotlib.pyplot as plt
# import seaborn as sns
# # sns.set(font_scale=1.2)
# # sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# # plt.show()

# # 도수 분포도
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 2, 1)
# plt.hist(x='sepal_length', data=df)
# plt.title('sepal_length')

# plt.subplot(2, 2, 2)
# plt.hist(x='sepal_width', data=df)
# plt.title('sepal_width')

# plt.subplot(2, 2, 3)
# plt.hist(x='petal_length', data=df)
# plt.title('petal_length')

# plt.subplot(2, 2, 4)
# plt.hist(x='petal_width', data=df)
# plt.title('petal_width')

# plt.show()