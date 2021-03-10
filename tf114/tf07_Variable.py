import tensorflow as tf
tf.compat.v1.set_random_seed(777)

W = tf.Variable(tf.compat.v1.random_normal([1]), name = 'weight')
print(W) # <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>
# sess = tf.Session()
sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer())
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(W)
print('aaa : ', aaa)
sess.close()

# sess = tf.InteractiveSession()
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = W.eval()
print('bbb : ', bbb)
sess.close()

sess= tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = W.eval(session=sess)
print('ccc : ', ccc)
sess.close()
