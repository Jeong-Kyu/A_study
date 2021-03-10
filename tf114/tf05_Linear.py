import tensorflow as tf
tf.set_random_seed(66)

x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias') # sess 통과전 variable초기화를 시켜야한다!

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())  
# print(sess.run(W), sess.run(b))

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) #loss -> mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for step in range(2001):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(cost), sess.run(W), sess.run(b))

for step in range(2001):
    sess.run(train)
    if step<=2:
        print("epoch : ",step)
        print("x : ",x_train)
        print("W : ",sess.run(W))
        print("b : ",sess.run(b))
        print("W*x + b = hypothesis , ",sess.run(hypothesis))
        print("y : ",y_train)
        print("hypothesis - y_train : ",sess.run(hypothesis - y_train))
        print("cost : ",sess.run(cost))
        print("\n\n")
# 1~3 계산과정 손으로 풀어보시오~

# Groupby
# 전체 데이터를 그룹 별로 나누고 (split), 각 그룹별로 집계함수를 적용(apply) 한후, 그룹별 집계 결과를 하나로 합치는(combine) 단계를 거치게 됩니다. (Split => Apply function => Combine)
