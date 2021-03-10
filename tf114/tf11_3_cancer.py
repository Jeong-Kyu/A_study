# accuracy_score
# r2_score
from sklearn.datasets import load_breast_cancer
import tensorflow as tf

dataset = load_breast_cancer()
x_data = dataset.data
y_data = dataset.target
y_data = y_data.reshape(-1,1)
print(x_data.shape)
print(y_data.shape)


x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None,1])


w = tf.Variable(tf.zeros([30,1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# cost = tf.reduce_mean(tf.square(hypothesis-y))
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.00000031).minimize(cost)

predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))

from sklearn.metrics import r2_score, accuracy_score

with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())
    for step in range(5001):    
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})

        if step % 200 == 0 :    
            print(step, cost_val)

    # h,c,a = sess.run([hypothesis,predicted,accuracy],feed_dict={x:x_data, y:y_data})
    # print("예측값 : ", h, "원래값 : ", c, "정확도", a)

    y_predict_value = sess.run(predicted, feed_dict={x: x_data,y:y_data})
    # print(y_predict_value, y_data)
    print("acc: ",accuracy_score(y_data,y_predict_value))

sess.close()

# adam (learning_rate=0.0000023)
# acc:  0.9050966608084359
# GradientDescentOptimizer(learning_rate=0.00000031)
# acc:  0.906854130052724