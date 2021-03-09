import tensorflow as tf

sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype = tf.float32, name = 'test') # sess 통과전 variable초기화를 시켜야한다!
# init = tf.global_variables_initializer()  #-------> 변수 초기화
init = tf.compat.v1.global_variables_initializer()  #-------> 변수 초기화
sess.run(init)
print(sess.run(x))