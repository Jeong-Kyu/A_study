# r2_score
from sklearn.datasets import load_wine
import tensorflow as tf
tf.set_random_seed(66)

dataset = load_wine()
x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(-1,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8)
print(x_data.shape) #(178, 13)
print(y_data.shape) #(178, 1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 원-핫 인코딩 적용
encoder = OneHotEncoder()
encoder.fit(y_test)
y_test = encoder.transform(y_test)
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test=y_test.toarray()
y_train=y_train.toarray()

x = tf.placeholder('float', [None, 13])
y = tf.placeholder('float', [None, 3])

w = tf.Variable(tf.random_normal([13, 3]), name = 'weight')
b = tf.Variable(tf.random_normal([1, 3]), name = 'bias')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# loss = tf.reduce_mean(tf.square(hypothesis-y)) #mse
from sklearn.metrics import r2_score, accuracy_score
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) #categorical_crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())

    for step in range(2001):    
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if step % 200 == 0 : 
            print(step, cost_val)
    
    a = sess.run(hypothesis, feed_dict={x:x_test})
    print("acc: ",accuracy_score(sess.run(tf.argmax(y_test,1)),sess.run(tf.argmax(a,1))))


# sess.close()
# AdamOptimizer(learning_rate=0.01)
# acc:  0.9722222222222222