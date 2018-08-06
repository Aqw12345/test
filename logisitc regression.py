# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:09:14 2018

@author: dk
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#from mnist import read_data_sets
import matplotlib.pyplot as plt
import pylab

mnist=input_data.read_data_sets(u"g:/mnist/MNIST_data/", one_hot=True)
#mnist=read_data_sets(u"g:/mnist/MNIST_data/", one_hot=True)
#print(mnist.train.images.shape,mnist.train.labels.shape)
#print(mnist.test.images.shape,mnist.test.labels.shape)
#print(mnist.validation.images.shape,mnist.validation.labels.shape)

#parameters
learning_rate=0.5
training_epoch=25
batch_size=100
display_step=1

# Input
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

#set model weight
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

#Construct model-softmat
pred=tf.nn.softmax(tf.matmul(x,W)+b)

#Minimize error using cross entropy
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))

#Gradient Descent
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
 
    #training cycle
    for epoch in range(training_epoch):
        avg_cost=0
        total_batch=int(mnist.train.num_examples/batch_size) #训练的量
        
        #Loop over all batches从训练集里一次提取batch_size张图片数据来训练，然后循环total_batch次，以达到训练的目的
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            #优化Run optimization op (backprop) and cost op (to get loss value)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            
            #compute averge loss
            avg_cost+=c/total_batch
        #display logs per epoch step
        if (epoch+1)%display_step==0:
            print("Epoch:",'%4d' % (epoch+1),"cost=","{:.9f}".format(avg_cost))
            
    print("Optimization Finished!")
    
    #Test model
    correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    ## Calculate accuracy
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:",sess.run(accuracy, feed_dict={x: mnist.test.images,y: mnist.test.labels}))
    
    #可视化
    for i in range(0, len(mnist.test.images)):
        result = sess.run(correct_prediction, feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]])})
        if not result:
            print('预测的值是：',sess.run(pred, feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]])}))
            print('实际的值是：',sess.run(y,feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]])}))
            one_pic_arr = np.reshape(mnist.test.images[i], (28, 28))
            pic_matrix = np.matrix(one_pic_arr, dtype="float")
            plt.imshow(pic_matrix)
            pylab.show()
            break
 
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,y: mnist.test.labels}))