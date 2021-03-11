import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_test_ = y_test.reshape(-1,1).copy()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

# 2. 모델 구성
x = tf.placeholder('float', [None,784])
y = tf.placeholder('float', [None,10])

w1 = tf.Variable(tf.random_normal([784,100]), name='weight1')
w1 = tf.get_variable('w1', shape=[784,100], initializer=tf.contrib.layers.xavier_initializer()) # kernel_initializer

print('w1 :', w1) # w1 : <tf.Variable 'w1:0' shape=(784, 100) dtype=float32_ref>

b1= tf.Variable(tf.random_normal([100]), name='bias1')

print('b1 :', b1) # b1 : <tf.Variable 'bias1:0' shape=(100,) dtype=float32_ref>

# layer1 = tf.nn.softmax(tf.matmul(x,w1) + b1)
# layer1 = tf.nn.relu(tf.matmul(x,w1) + b1)
# layer1 = tf.nn.selu(tf.matmul(x,w1) + b1)
layer1 = tf.nn.relu(tf.matmul(x,w1) + b1)

print('layer1 :', layer1) # layer1 : Tensor("Elu:0", shape=(?, 100), dtype=float32)

layer1 = tf.nn.dropout(layer1, keep_prob=0.3) # dropout(0.3)

print('layer1 :', layer1) # layer1 : Tensor("dropout/mul_1:0", shape=(?, 100), dtype=float32)

w2 = tf.get_variable('weight2', shape=[100,128], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([128]), name='bias2')
layer2 = tf.nn.elu(tf.matmul(layer1,w2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.3) # dropout(0.3)

w3 = tf.get_variable('weight3', shape=[128,64], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([64]), name='bias3')
layer3 = tf.nn.selu(tf.matmul(layer2,w3) + b3)
layer3 = tf.nn.dropout(layer3, keep_prob=0.3) # dropout(0.3)

w4 = tf.get_variable('weight4', shape=[64,10], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([10]), name='bias4')
hypothesis = tf.nn.softmax(tf.matmul(layer3,w4) + b4)


# 3. 컴파일, 훈련
loss = tf.reduce_mean(- tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


# Batch_size 설정
batch_size = 100
total_batch = int(len(x_train) / batch_size)     # 600
training_epochs = 15

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]

        feed_dict = {x:batch_x, y:batch_y}
        
        c, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

        avg_cost += c / total_batch

    print('Epoch :', '%04d' %(epoch + 1),
          'cost = {:.9f}'.format(avg_cost))

    if epoch == 8:
        break

print('Finish !!!')

prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc :', sess.run(accuracy, feed_dict={x:x_test[:10], y:y_test[:10]}))




y_true= tf.placeholder(tf.float32, shape=[None,1])

for count, (i, j) in enumerate(zip(sess.run(y_true, feed_dict={y_true:y_test_}), sess.run(tf.arg_max(hypothesis, 1), feed_dict={x:x_test}))):
    print('y_true :', i, "\ty_pred :", j)
    if count == 10:
        break