#즉시 실행 모드
# from tensorflow.python.framework.ops import disable_eager_execution 
import tensorflow as tf
print(tf.executing_eagerly()) # False
tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly()) # False

print(tf.__version__) #2.3.1

hello = tf.constant("Hello World")
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello))
