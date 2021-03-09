# boston, diabets, cancer, iris, wine
# mnist, fashion, cifar10, cifar100

from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer, load_iris, load_wine
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np

# 1. boston
boston_dataset = load_boston()

boston_x = boston_dataset.data
boston_y = boston_dataset.target

np.save('../data/npy/boston_x.npy', arr=boston_x)
np.save('../data/npy/boston_y.npy', arr=boston_y)
# 2. diabets
diabetes_dataset = load_diabetes()

diabetes_x = diabetes_dataset.data
diabetes_y = diabetes_dataset.target

np.save('../data/npy/diabetes_x.npy', arr=diabetes_x)
np.save('../data/npy/diabetes_y.npy', arr=diabetes_y)
# 3. cancer
cancer_dataset = load_breast_cancer()

cancer_x = cancer_dataset.data
cancer_y = cancer_dataset.target

np.save('../data/npy/cancer_x.npy', arr=cancer_x)
np.save('../data/npy/cancer_y.npy', arr=cancer_y)
# 4. iris
iris_dataset = load_iris()

iris_x = iris_dataset.data
iris_y = iris_dataset.target

np.save('../data/npy/iris_x.npy', arr=iris_x)
np.save('../data/npy/iris_y.npy', arr=iris_y)
# 5. wine
wine_dataset = load_wine()

wine_x = wine_dataset.data
wine_y = wine_dataset.target

np.save('../data/npy/wine_x.npy', arr=wine_x)
np.save('../data/npy/wine_y.npy', arr=wine_y)


# 6. mnist
(m_x_train, m_y_train), (m_x_test, m_y_test) = mnist.load_data()
np.save('../data/npy/mnist_x_train.npy', arr = m_x_train)
np.save('../data/npy/mnist_x_test.npy', arr = m_x_test)
np.save('../data/npy/mnist_y_train.npy', arr = m_y_train)
np.save('../data/npy/mnist_y_test.npy', arr = m_y_test)
# 7. fashion
(f_x_train, f_y_train), (f_x_test, f_y_test) = fashion_mnist.load_data()
np.save('../data/npy/fashion_mnist_x_train.npy', arr = f_x_train)
np.save('../data/npy/fashion_mnist_x_test.npy', arr = f_x_test)
np.save('../data/npy/fashion_mnist_y_train.npy', arr = f_y_train)
np.save('../data/npy/fashion_mnist_y_test.npy', arr = f_y_test)
# 8. cifar10
(c10_x_train, c10_y_train), (c10_x_test, c10_y_test) = cifar10.load_data()
np.save('../data/npy/cifar10_x_train.npy', arr = c10_x_train)
np.save('../data/npy/cifar10_x_test.npy', arr = c10_x_test)
np.save('../data/npy/cifar10_y_train.npy', arr = c10_y_train)
np.save('../data/npy/cifar10_y_test.npy', arr = c10_y_test)
# 9. cifar100
(c100_x_train, c100_y_train), (c100_x_test, c100_y_test) = cifar100.load_data()
np.save('../data/npy/cifar100_x_train.npy', arr = c100_x_train)
np.save('../data/npy/cifar100_x_test.npy', arr = c100_x_test)
np.save('../data/npy/cifar100_y_train.npy', arr = c100_y_train)
np.save('../data/npy/cifar100_y_test.npy', arr = c100_y_test)
