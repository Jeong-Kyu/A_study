import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def Leaky_relu(x):
    return np.maximum(0.01*x, x)

def elu(x, alp):
    return (x>0)*x + (x<=0)*(alp*(np.exp(x)-1))

x = np.arange(-5,5,0.1)
y = elu(x,1)

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()


## 과제
# elu, selu, reaky relu