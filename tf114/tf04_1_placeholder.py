import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)
node3 = tf.add(node1, node2)

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5})) # 7.5
print(sess.run(adder_node, feed_dict={a:[1,3], b:[3,4]})) # [4. 7.]

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a:4, b:2})) # 18.0