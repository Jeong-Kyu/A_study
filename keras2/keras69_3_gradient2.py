import numpy as np

f = lambda x : x**2 - 4*x + 6

gradient = lambda x : 2*x - 4

x0 = 10.0 # 랜덤값
epoch = 100
learning_rate = 0.5

print('step\tx\tf(x)')
print('{:02d}\t{:6.5f}\t{:6.5f}'.format(0, x0, f(x0)))

for i in range(epoch):
    temp = x0 - learning_rate * gradient(x0)  #원값 - 런닝레이트 * 미분값 -> 최적의 w를 찾는 과정
    x0 = temp
    print('{:02d}\t{:6.5f}\t{:6.5f}'.format(i+1, x0, f(x0)))