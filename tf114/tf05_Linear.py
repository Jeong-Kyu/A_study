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

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))