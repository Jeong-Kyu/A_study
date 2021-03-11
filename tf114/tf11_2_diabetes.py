# r2_score
from sklearn.datasets import load_diabetes
import tensorflow as tf

dataset = load_diabetes()
x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(-1,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8)
print(x_data.shape)
print(y_data.shape)


x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None,1])


w = tf.Variable(tf.random_normal([10,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


hypothesis = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.99)
train = optimizer.minimize(cost)

from sklearn.metrics import r2_score
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())
    for step in range(200001):   
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_train, y:y_train})
        if step % 10 ==0:
            print(step, "cost : ", cost_val)#, "\n", hy_val)

    y_predict_value = sess.run(hypothesis, feed_dict={x: x_test})
    print("r2: ",r2_score(y_test,y_predict_value))

sess.close()

# r2:  0.5144020896348773 AdamOptimizer(learning_rate=0.99999999)
# r2:  0.6277470615143084 GradientDescentOptimizer(learning_rate=0.99)