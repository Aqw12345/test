# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:13:44 2018

@author: dk
"""
#sample neuron network (无激活函数)
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#from mnist import read_data_sets
import numpy as np
import pylab
import matplotlib.pyplot as plt

mnist=input_data.read_data_sets(u"g:/mnist/MNIST_data/",one_hot=True)

#parameters
learning_rate=0.1
num_steps=500
batch_size=128
display_step=100

#Network paraments-1st,2st layer number of neurons
n_hidden_1=256 
n_hidden_2=256
num_input=784 #mnist 28*28 feature
num_classes=10 #output labels(0-9)

x=tf.placeholder(tf.float32,[None,num_input])
y=tf.placeholder(tf.float32,[None,num_classes])

#weight and bias
weight={'h1':tf.Variable(tf.random_normal([num_input,n_hidden_1])),\
        'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),\
        'out':tf.Variable(tf.random_normal([n_hidden_2,num_classes]))}

biases={'b1':tf.Variable(tf.random_normal([n_hidden_1])),\
        'b2':tf.Variable(tf.random_normal([n_hidden_2])),\
        'out':tf.Variable(tf.random_normal([num_classes]))}

#create model
def neuron_net(x):
    #Hidden1 fully connected layer with 256 neurons
    layer_1=tf.add(tf.matmul(x,weight['h1']),biases['b1'])
    #Hidden2 fully connected layer with 256 neurons
    layer_2=tf.add(tf.matmul(layer_1,weight['h2']),biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer=tf.add(tf.matmul(layer_2,weight['out']),biases['out'])
    return out_layer

#Construct model
logits=neuron_net(x)
prediction=tf.nn.softmax(logits)

#loss function and optimizer
loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

#Evaluate model
correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1,num_steps+1):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if (step%display_step==0)or(step==1):
            loss,acc=sess.run([loss_op,accuracy],feed_dict={x:batch_x,y:batch_y})
            print("Step:"+ str(step)+",loss="+"{:.4f}".format(loss)+",training accuracy="+"{:.3f}".format(acc))
    print("Optimization Finished!")
    
    print("Testing Accuracy:",sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

    #可视化
    for i in range(0, len(mnist.test.images)):
        result = sess.run(correct_pred, feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]])})
        if not result:
            print('预测的值是：',sess.run(prediction, feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]])}))
            print('实际的值是：',sess.run(y,feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]])}))
            one_pic_arr = np.reshape(mnist.test.images[i], (28, 28))
            pic_matrix = np.matrix(one_pic_arr, dtype="float")
            plt.imshow(pic_matrix)
            pylab.show()
            break
    
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,y: mnist.test.labels}))