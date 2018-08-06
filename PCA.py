# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:19:43 2018

@author: dk
"""

import numpy as np


#生成2个3*20的数据集,行表示其维度
np.random.seed(2356889598)

mu_vec1=np.array([0,0,0])
cov_mat1=np.array([[1,0,0],[0,1,0],[0,0,1]])
#根据实际情况生成一个多元正太分布矩阵
class1_sample=np.random.multivariate_normal(mu_vec1,cov_mat1,20).T ##依据指定的均值和协方差生成数据，shape=(20,3) 3——mean的长度，即mu_vecl

#python assert断言是声明其布尔值必须为真的判定，如果发生异常就说明表达示为假。可以理解assert断言语句为raise-if-not，用来测试表示式，其返回值为假，就会触发异常。
#assert+空格+要判断语句+双引号“报错语句”
#如果你断言的 语句正确 则什么反应都没有 但是如果你出错之后 就会报出    AssertionError 并且错误可以自己填写，即双引号中的内容
assert class1_sample.shape==(3,20),"The matrix has not the dimensions 3*20"

mu_vec2=np.array([1,1,1])
cov_mat2=np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample=np.random.multivariate_normal(mu_vec2,cov_mat2,20).T
assert class2_sample.shape==(3,20),"The matrix has not the dimensions 3*20"

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(111,projection='3d')
plt.rcParams['legend.fontsize']=10
ax.plot(class1_sample[0,:],class1_sample[1,:],class1_sample[2,:],'o',markersize=8,color='blue',alpha=0.5,label='class1')
ax.plot(class2_sample[0,:],class2_sample[1,:],class2_sample[2,:],'^',markersize=8,color='red',alpha=0.5,label='class2')

plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')
plt.show()


#PCA . 1.取除了标签外的所有维度:因为PCA分析不需要类别标签，所以我们把两个类别的样本合并起来形成一个3×40的一个矩阵
all_samples=np.concatenate((class1_sample,class2_sample),axis=1) #列增加，按行合并
assert all_samples.shape==(3,40),"The matrix has not the dimensions 3x40"

#2.计算一个d维均值向量，离均值化
mean=np.mean(all_samples,axis=1) #按行求均值，行表示维度
mean=mean[:,np.newaxis]
newData=all_samples-mean

#求协方差矩阵
cov_mat=np.cov(newData)

#求特征值和特征向量
eigenvalues,eigenvector=np.linalg.eig(np.mat(cov_mat))

#对特征向量按照特征值降序排列
eigValIndice=np.argsort(eigenvalues)   #对特征值从小到大排序
n_eigValIndice=eigValIndice[-1:-3:-1] #最大的n个特征值的下标 ,-(n+1),此时n=2
n_eigVect=eigenvector[n_eigValIndice,:]  ##最大的n个特征值对应的特征向量 
lowDate=n_eigVect.dot(newData) #降为后的新数据

#重构数据，即原始数据
reconMat=(n_eigVect.T.dot(lowDate))


#降为之后的可视化
plt.plot(lowDate[0,0:20], lowDate[1,0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(lowDate[0,20:40], lowDate[1,20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples with class labels')

plt.show()