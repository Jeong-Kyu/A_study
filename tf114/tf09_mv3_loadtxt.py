import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

dataset = np.loadtxt('C:\data\csv\data-01-test-score.csv',delimiter=',')
data_pred = dataset[:5,:]
dataset = dataset[:5,:]
x_data = dataset[:,:3]
y_data = dataset[:,3].reshape(-1,1)
x_pred = data_pred[:,:3]
y_pred = data_pred[:,3].reshape(-1,1)

print(x_data.shape) #(25, 3)
print(y_data.shape) #(25, )


x = tf.placeholder(tf.float32, shape=[None,3])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


hypothesis = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):   
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})
    if step % 10 ==0:
        print(step, "cost : ", cost_val, "\n", hy_val)
    print(sess.run(hypothesis, feed_dict={x:x_pred, y:y_pred}))
sess.close()