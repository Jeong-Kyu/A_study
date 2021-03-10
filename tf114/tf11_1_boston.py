# r2_score
from sklearn.datasets import load_boston
import tensorflow as tf

dataset = load_boston()
x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(-1,1)
print(x_data.shape)
print(y_data.shape)


x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None,1])


w = tf.Variable(tf.random_normal([13,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


hypothesis = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.9)
train = optimizer.minimize(cost)

from sklearn.metrics import r2_score
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())
    for step in range(2001):   
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})
        if step % 10 ==0:
            print(step, "cost : ", cost_val)#, "\n", hy_val)

    y_predict_value = sess.run(hypothesis, feed_dict={x: x_data})
    print("r2: ",r2_score(y_data,y_predict_value))

sess.close()