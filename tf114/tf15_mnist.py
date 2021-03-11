import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],784)/255.
x_test = x_test.reshape(x_test.shape[0],784)/255.
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

tf.set_random_seed(66)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(y_test)
y_test = encoder.transform(y_test)
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test=y_test.toarray()
y_train=y_train.toarray()

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

# w = tf.Variable(tf.random_normal([784,10]),name = 'weight1')
# b = tf.Variable(tf.random_normal([10]),name = 'bias1')
# hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

w1 = tf.Variable(tf.random_normal([784,10],stddev=0.1),name = 'weight1')
b1 = tf.Variable(tf.random_normal([10],stddev=0.1),name = 'bias1')
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([10,10],stddev=0.1),name = 'weight2')
b2 = tf.Variable(tf.random_normal([10],stddev=0.1),name = 'bias2')
# layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
hypothesis = tf.nn.softmax(tf.matmul(layer1, w2) + b2)

# w3 = tf.Variable(tf.random_normal([10, 10]),name = 'weight3')
# b3 = tf.Variable(tf.random_normal([10]),name = 'bias3')
# hypothesis = tf.nn.softmax(tf.matmul(layer2, w3) + b3)

# loss = tf.reduce_mean(tf.square(hypothesis-y)) #mse

from sklearn.metrics import r2_score, accuracy_score
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) #categorical_crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())

    for step in range(2001):    
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if step % 200 == 0 : 
            print(step, cost_val)
    
    a = sess.run(hypothesis, feed_dict={x:x_test})
    print("acc: ",accuracy_score(sess.run(tf.argmax(y_test,1)),sess.run(tf.argmax(a,1))))

# acc:  0.8613 AdamOptimizer(learning_rate=0.001)
# acc:  0.8833 AdamOptimizer(learning_rate=0.0015)
# acc:  0.9322 AdamOptimizer(learning_rate=0.001) + layer 1