import tensorflow as tf
from tensorflow.python.framework import dtypes
a=3
#create a variable
# 首先要把变量、操作写好，然后进行全局初始化，构造session计算区域，
w = tf.Variable([[0.5, 1.0]])
x = tf.Variable([[2.0], [1.0]])
y = tf.matmul(w, x)
print(y)

init_op = tf.global_variables_initializer()
with tf.Session() as sess: # 具体的值放在session图里面然后在这个图里进行计算
    sess.run(init_op)
    print(y.eval()) # eval:具体的值

# 基本的数据结构
tf.zeros([3,4], dtype=float) # 3行4列的0

tensor = tf.Variable([[1,2,3],[4,5,6]])
tf.zeros_like(tensor) #类似tensor结构，全部是0
tensor = tf.ones([2,3], dtype=dtypes.float32) #[[1,1,1,],[1,1,1]]两行三列全是1

tf.ones_like(tensor) #类似zeros_like 两行三列全是1

tensor = tf.constant([1,2,3,4,5,6,7]) #[1,2,3,4,5,6,7]

tensor = tf.constant(-1.0, shape=[2,3]) #[[-1,-1,,-1],[-1,-1,-1]]

tf.linspace(10.0, 12.0, 3,name='linspace') #[初始值， 结束值，总共个数]
tf.range(start=3, limit=18,delta=3, name='raange') #[3,6,9,12,,15]

norm = tf.random_normal([2,3], mean=-1, stddev=4) # mean:均值，stddev：指定方差，求高斯分布

c = tf.constant([[1,2], [3,4], [5,6]])
shuff = tf.random_shuffle(c) #对原始数据进行洗牌操作，比如成[[1,2],  [5,6], [3,4]]

state = tf.Variable(0)
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value) # new_value赋值给state

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))
    for _ in range(3):
        print(sess.run(update))

# 将session随时保存
w = tf.Variable([[0.5, 1.0]])
x = tf.Variable([[2.0], [1.0]])
y = tf.matmul(w, x)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    save_path = saver.save(sess, "./save/model/test")
    print('model saved in file:', save_path)

# numpy转化成tf
import numpy as np
a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
    print(sess.run(ta))

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2) #multiply点乘 matmul矩阵乘法
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]})) #这种结构，可以在run时候进行赋值




















