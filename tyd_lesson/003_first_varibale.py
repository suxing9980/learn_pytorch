import tensorflow as tf
a=3
#create a variable
w = tf.Variable([[0.5, 1.0]])
x = tf.Variable([[2.0], [1.0]])
y = tf.matmul(w, x)
print(y)

init_op = tf.global_variables_initializer()
with tf.Session() as sess: # 具体的值放在session图里面然后在这个图里进行计算
    sess.run(init_op)
    print(y.eval()) # eval:具体的值