#!/usr/bin/env python
# -*- coding:utf-8-*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/', one_hot=True)

n_hidden_1 = 256 # 第一层有多少神经元
n_hidden_2 = 128 # 第二层有多少神经元
n_input = 784
n_classes = 10 #最终分类的类别

# inputs and outputs
x = tf.placeholder('float', [None, n_input]) #多少个样本不知道，每一个样本是由784个像素点构成的
y = tf.placeholder('float', [None, n_classes])

# network paramerters
stddev = 0.1
weights = {
    'w1' : tf.Variable(tf.random_normal([n_input, n_hidden_1] , stddev=stddev)), # 高斯初始化
    'w2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2] , stddev=stddev)),
    'out' : tf.Variable(tf.random_normal([n_hidden_2, n_classes] , stddev=stddev)),
}
biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}
print('network ready!')

def multilayer_perceptron(_X, _weights, _biases): #前向传播，首先是data
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']),_biases['b1'])) #每一层计算完了，加sigmoid激活函数 w*x + b
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['w2']),_biases['b2']))
    return (tf.matmul(layer_2, _weights['out']) + _biases['out'])
#prediction
pred = multilayer_perceptron(x, weights, biases)
#loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(pred, y)) #交叉熵函数，损失函数 第一个输入是网络预测值 第二个是实际label值 reduce_mean求的是平均loss
optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) #corr准确率
accr = tf.reduce_mean(tf.cast(corr, 'float')) # cast转换成float类型  完成float之后就计算出精度

init = tf.global_variables_initializer()
training_epochs = 20
batch_size = 100
display_step = 4
# lauch the craph
sess = tf.Session(init)
#optimize
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    #iteration
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x:batch_xs, y:batch_ys}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = avg_cost / total_batch
    # display
    if(epoch + 1) % display_step == 0:
        print('Epoch:%03d/%03d cost :%.9f' %(epoch, training_epochs, avg_cost))
        feeds = {x:batch_xs, y:batch_ys}
        train_acc = sess.run(accr, feed_dict=feeds)
        print('train accuracy:%.3f' % (train_acc))
        feeds = {x:mnist.test.images, y:mnist.test.labels}
        test_acc = sess.run(accr, feed_dict=feeds)
        print('test accuracy:%.3f' % (test_acc))
print('optimization finished!')