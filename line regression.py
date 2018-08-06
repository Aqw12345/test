# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:13:15 2018

@author: dk
"""
#第一种方式

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#train data
train_X=np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y=np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])

# number of train_X data
n_sample=train_X.shape[0]

#parameters
#设置学习率,设置训练步数,设置结果显示步数
learning_rate=0.01
training_epochs=1000
display_step=50

# X Y的占位符,设置成32位浮点数
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

#设置随机权重(weight),设置偏差(bias)为零
w=tf.Variable(tf.random_uniform([1]))
b=tf.Variable(tf.zeros([1]))

# 构造线性模型  y = x*w + b
pred=tf.add(tf.multiply(X,w),b)

# 计算均方误差  cost fucation
cost=tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_sample)

#梯度下降 Gradient descent
# Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#初始化全部变量
init=tf.global_variables_initializer()

#start training
with tf.Session() as sess:
    #Run the initializer
    sess.run(init)
    #Fit all training data
    for epoch in range(training_epochs):
        sess.run(optimizer,feed_dict={X:train_X,Y:train_Y})
        #Display logs per epoch step
        if (epoch+1) % display_step==0:
            c=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            #一种格式化字符串的函数str.format()
            print("Epoch:",'%04d' % (epoch+1),"cost=","{:.9f}".format(c),\
                  "w=",sess.run(w),"b=",sess.run(b))
            
    print("Optimization Finished!")
    training_cost=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
    print("Training cost=",training_cost,"w=",sess.run(w),"b=",sess.run(b),'\n')

    #Graphic display
    plt.plot(train_X,train_Y,'ro',label='Original data')
    plt.plot(train_X,sess.run(w)*train_X+sess.run(b),label='Fitted line')
    plt.legend()
    plt.show()
    
    #Testing example,as requested(Issue #2)测试训练方案，在测试集上的准确性
    test_X=np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y=np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
    
    print("Testing... (Mean square loss Comparison)")
    testing_cost=sess.run(
            tf.reduce_sum(tf.pow(pred-Y,2))/(2*test_X.shape[0]),feed_dict={X:test_X,Y:test_Y}) #same function as cost above
    print("Testing cost=",testing_cost)
    print("Absolute mean square loss difference:",abs(training_cost-testing_cost))
    
    plt.plot(test_X,test_Y,'bo',label='Testing date')
    plt.plot(train_X,sess.run(w)*train_X+sess.run(b),label='Fitted line')
    plt.legend()
    plt.show()
    
'''
#利用tensorboard，第二种方式
import tensorflow as tf
import numpy as np

N=100
w_true=5
b_true=2
noise_scale=0.1
x_np=np.random.rand(N,1)
noise=np.random.normal(scale=noise_scale,size=(N,1))
y_np=np.reshape(w_true*x_np+b_true+noise,(-1))
n_sample=x_np.shape[0]
with tf.name_scope("placeholder"):
    x=tf.placeholder(tf.float32)
    y=tf.placeholder(tf.float32)

with tf.name_scope("Weights"):
    W=tf.Variable(tf.random_normal([1]))
    b=tf.Variable(tf.random_normal([1]))
    
with tf.name_scope("predicton"):
    y_pred=tf.matmul(x,W)+b
    
with tf.name_scope("loss"):
    l=tf.reduce_sum(tf.pow(y-y_pred,2))/(2*n_sample)
    
with tf.name_scope("optim"):
    train_op=tf.train.GradientDescentOptimizer(0.001).minimize(l)
    
with tf.name_scope("summaries"):
    tf.summary.scalar("loss",l)
    merged=tf.summary.merge_all()
    
train_writer=tf.summary.FileWriter('g:\lr-train',tf.get_default_graph())
n_steps=1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(n_steps):
        optim,summary,loss=sess.run([train_op,merged,l],feed_dict={x:x_np,y:y_np})
        print("step %d ,loss: %f"%(i,loss))
        train_writer.add_summary(summary,i)
'''