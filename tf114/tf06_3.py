# tf06_2.py lr수정
# epoch <2000

# placeholder

import tensorflow as tf
tf.set_random_seed(66)

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

# x_train = [1,2,3]
# y_train = [3,5,7]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.172433).minimize(cost)  # learning_rate = 0.1725 [1.9984291] [1.0042138]  learning_rate = 0.172433 [1.9984076] [1.0042126]

with tf.Session() as sess:            #with문을 통해 sess.close() 없이도 적용
    sess.run(tf.global_variables_initializer())
    for step in range(101):
        _, cost_val, W_val, b_val = sess.run([train,cost,W,b],feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
    print(sess.run(hypothesis, feed_dict={x_train:[4]}))
    print(sess.run(hypothesis, feed_dict={x_train:[5,6]}))
    print(sess.run(hypothesis, feed_dict={x_train:[6,7,8]}))

    
# 1.[4]
# 2.[5,6]
# 3.[6,7,8]

# [8.999977]
# [10.999964 12.99995 ]
# [12.99995  14.999937 16.999924]
