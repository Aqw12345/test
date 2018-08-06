# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:19:35 2018

@author: dk
"""

import numpy as np

####--------------------------数据的行表示维度，即特征，列表示样本
def zeroMean(data):
    mean=np.mean(data)
    newDate=data-mean
    return newDate,mean

#选择主成分个数,根据公式，可以写个函数，函数传入的参数是百分比percentage和特征值向量，然后根据percentage确定n
def percentage2n(eigVals,percentage):
    sortArray=np.sort(eigVals) #升序
    sortArray=sortArray[-1::-1] #降序
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num
        

#---------------------------------------------------PCA
def pca(data,percentage):
    #零均值化
    newDate,mean=zeroMean(data)
    #协方差
    cov_mat=np.cov(newDate)
    #求特征值和特征向量
    eigVals,eigVects=np.linalg.eig(cov_mat)
    #求n
    n=percentage2n(eigVals,percentage)
    #求相应的降为数据
    eigVaIndice=np.argsort(eigVals) #对特征值从小到大排序
    n_eigVaIndice=eigVaIndice[-1:-(n+1):-1] #最大的n个特征值的下标
    n_eigVect=eigVects[n_eigVaIndice,:]  #最大的n个特征值对应的特征向量 
    lowDate=n_eigVect.dot(newDate)
    reconDate=(n_eigVect.T.dot(lowDate))+mean
    return lowDate,reconDate

#-----------------------------------------------------------
np.random.seed(2356889598)

mu_vec1=np.array([0,0,0])
cov_mat1=np.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample=np.random.multivariate_normal(mu_vec1,cov_mat1,20).T ##依据指定的均值和协方差生成数据，shape=(20,3) 3——mean的长度，即mu_vecl

mu_vec2=np.array([1,1,1])
cov_mat2=np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample=np.random.multivariate_normal(mu_vec2,cov_mat2,20).T

all_samples=np.concatenate((class1_sample,class2_sample),axis=1) #列增加，按行合并

cc=pca(all_samples,0.6)
