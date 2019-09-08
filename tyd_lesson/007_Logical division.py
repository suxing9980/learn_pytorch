#!/usr/bin/env python
# -*- coding:utf-8-*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/data/', one_hot=True) #不光下载数据，同时对以下几组数据做了预处理
# 对数据进行切分
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabels = mnist.test.labels
print('MNIST loaded')

print(trainimg.shape) #28*28 784个像素点 n*784 n个样本数，每个数据集28*28像素大小
print(trainlabel.shape)
print(testimg.shape)
print(testlabels.shape)
print(trainlabel[0])
print(trainlabel[9])

# 下来做逻辑回归这个事
x = tf.placeholder('float', [None, 784]) #占位符 不知道多大，None表示无穷，784像素点,不知道多少样本,行数不知道
y = tf.placeholder('float', [None, 10])  #784与10是列填充
W = tf.Variable(tf.zeros([784, 10])) #能相乘，w当做行向量，x当做列向量，所以w也得是行784
b = tf.Variable(tf.zeros(10))#只是10分类。初始化10个b就可以 这里是零值初始化，一般选高斯初始化很不错
# LOGISTIC REGRESSION MODEL
actv = tf.nn.softmax(tf.matmul(x, W) + b) # softmax多分类，在softmax模型下计算当前模型 softmax传递这么一个分值 建造这么一个模型
#COST FUNCTION
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), reduction_indices=1)) #逻辑回归损失函数，-log（p）
# OPTIMIZER
learning_rate = 0.01
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #梯度下降就优化
# 模型搭建好了来测试
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1)) # 预测值索引与真实label值索引是不是一样,1表示按照行求,返回true与false
accr = tf.reduce_mean(tf.cast(pred, "float")) #true与false转行成0与1，精确 所有值加在一起就均值，0与1
# INITIALIZER
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
arr = np.array([[31,23,4,24,27,34],
                [18,3,25,0,6,35],
                [28,14,33,22,20,8],
                [13,30,21,19,7,9],
                [16,1,26,32,2,29],
                [17,12,5,11,10,15]])
print(tf.rank(arr).eval()) #rank查看维度 2维
print(tf.shape(arr).eval()) #几行几列 6行6列
print(tf.argmax(arr, 0).eval()) #返回最大值的索引 每一个列上最大值的索引 变成1就是按照行求最大

training_epochs =50 #所有样本迭代一次
batch_size =100     #每进行一次迭代选择多少样本
display_step =5     #展示
# SESSION
sess = tf.Session()
sess.run(init)
#MINI-BATCH LEARNING
for epoch in range(training_epochs):
    avg_cost = 0 #先让损失值等于0
    num_batch = int(mnist.train.num_examples/batch_size)
    for i in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) #一个是data、一个是label，一步步返回
        sess.run(optm, feed_dict={x:batch_xs, y:batch_ys}) #run的梯度下降求解
        feeds = {x:batch_xs, y:batch_ys}
        avg_cost += sess.run(cost, feed_dict=feeds)/num_batch #run损失函数
    # 打印当前的效果
    if(epoch % display_step == 0):
        feeds_train = {x:batch_xs, y:batch_ys}
        feeds_test = {x:mnist.test.images, y:mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print('Epoch: %03d/%03d cost:%.9f train_acc: %.3f test_acc: %.3f' % (epoch, training_epochs, avg_cost, train_acc,test_acc))
print('Done')







