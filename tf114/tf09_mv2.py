import tensorflow as tf
tf.set_random_seed(66)

x_data = [[73, 80, 75],[93, 88, 93], 
          [85, 91, 90],[96, 98, 100],[73, 66, 70]]
y_data = [[152],[185],[180],[196],[142]]

# 스칼라 백터 행렬 텐서 // 차원 내용

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

sess.close()