import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

print('Download and Extract MNIST dataset!')
mnist  = input_data.read_data_sets('data/', one_hot=True) #编码格式是01编码的

print('type of mnist is ',type(mnist))
print('number of train data is ',mnist.train.num_examples)
print('number of test data is ',mnist.test.num_examples)

# 对数据进行切分
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabels = mnist.test.labels

print('type of trainimg ',type(trainimg))
print('type of trainlabel ',type(trainlabel))
print('type of testimg ',type(testimg))
print('type of testlabels ',type(testlabels))

print('shape of trainimg ',trainimg.shape) #一个样本多少个像素点
print('shape of trainlabel ',trainlabel.shape) #每个样本有10个label，10分类
print('shape of testimg ',testimg.shape)
print('shape of trainlabel ',trainlabel.shape)
