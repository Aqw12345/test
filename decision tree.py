# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 13:36:23 2018

@author: dk
"""
from math import log
import operator

#计算数据的熵(entropy) H(X)
def calcShannonEnt(dataSet):
    numEntries=len(dataSet) # 数据条数
    labelCounts={}
    for featVec in dataSet: # 遍历每个实例，统计标签的频数
        currentLabel=featVec[-1] ## 每行数据的最后一个字（类别）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1  # 统计有多少个类以及每个类的数量
    shannonEnt=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries # 计算单个类的熵值
        shannonEnt-=prob*log(prob,2) #累加每个类的熵值
    return shannonEnt

#计算条件熵 H(Y|X)
def splitDataSet(dataSet, axis, value): # 按某个特征分类后的数据
    '''
    按照给定特征划分数据集
    :param dataSet:待划分的数据集
    :param axis:划分数据集的特征
    :param value: 需要返回的特征的值
    :return: 划分结果列表
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def calcConditionalEntropy(dataSet, i, featList, uniqueVals):
    '''
    计算X_i给定的条件下，Y的条件熵
    :param dataSet:数据集
    :param i:维度i
    :param featList: 数据集特征列表
    :param uniqueVals: 数据集特征集合
    :return: 条件熵
    '''
    conditionEnt = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)
        prob = len(subDataSet) / float(len(dataSet))  # 极大似然估计概率
        conditionEnt += prob * calcShannonEnt(subDataSet)  # 条件熵的计算
    return conditionEnt

#信息增益（information gain） g(D,A)
def calcInformationGain(dataSet, baseEntropy, i):
    '''
    计算信息增益
    :param dataSet:数据集
    :param baseEntropy:数据集的信息熵
    :param i: 特征维度i
    :return: 特征i对数据集的信息增益g(D|X_i)
    '''
    featList = [example[i] for example in dataSet]  # 第i维特征列表
    uniqueVals = set(featList)  # 转换成集合
    newEntropy = calcConditionalEntropy(dataSet, i, featList, uniqueVals)
    infoGain = baseEntropy - newEntropy  # 信息增益，就yes熵的减少，也就yes不确定性的减少
    return infoGain

#信息增益比（information gain ratio） gR(D,A)
def calcInformationGainRatio(dataSet, baseEntropy, i):
    '''
    计算信息增益比
    :param dataSet:数据集
    :param baseEntropy:数据集的信息熵
    :param i: 特征维度i
    :return: 特征i对数据集的信息增益比gR(D|X_i)
    '''
    return calcInformationGain(dataSet, baseEntropy, i) / baseEntropy

#导入数据集
def createDataSet():
    dataSet = [['youth', 'no', 'no', 1, 'refuse'],
               ['youth', 'no', 'no', '2', 'refuse'],
               ['youth', 'yes', 'no', '2', 'agree'],
               ['youth', 'yes', 'yes', 1, 'agree'],
               ['youth', 'no', 'no', 1, 'refuse'],
               ['mid', 'no', 'no', 1, 'refuse'],
               ['mid', 'no', 'no', '2', 'refuse'],
               ['mid', 'yes', 'yes', '2', 'agree'],
               ['mid', 'no', 'yes', '3', 'agree'],
               ['mid', 'no', 'yes', '3', 'agree'],
               ['elder', 'no', 'yes', '3', 'agree'],
               ['elder', 'no', 'yes', '2', 'agree'],
               ['elder', 'yes', 'no', '2', 'agree'],
               ['elder', 'yes', 'no', '3', 'agree'],
               ['elder', 'no', 'no', 1, 'refuse'],
               ]
    labels = ['age', 'working?', 'house?', 'credit_situation']  #特征
    return dataSet, labels


#ID3
def chooseBestFeatureToSplitByID3(dataSet):  # 选择最优的分类特征,即最大的信息熵，以及对应的特征
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)  # 原始的熵
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob =len(subDataSet)/float(len(dataSet))
            newEntropy +=prob*calcShannonEnt(subDataSet)  # 按特征分类后的熵
        infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
        if (infoGain>bestInfoGain):   # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            bestInfoGain=infoGain
            bestFeature = i
    return bestFeature

#采用多数表决的方法决定叶结点的分类，param: 所有的类标签列表  return: 出现次数最多的类
def majorityCnt(classList):    #按分类后类别数量排序，比如：最后分类为2男1女，则判定为男；
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#创建决策树    param: dataSet:训练数据集   return: labels:所有的类标签
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]  # 类别：男或女
    if classList.count(classList[0])==len(classList):
        return classList[0]   # 第一个递归结束条件：所有的类标签完全相同
    if len(dataSet[0])==1:
        return majorityCnt(classList)   # 第二个递归结束条件：用完了所有特征
    #bestFeat=chooseBestFeatureToSplitByc(dataSet) #选择最优特征ID3
    bestFeat=chooseBestFeatureToSplitByC45(dataSet) #选择最优特征C4.5
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}        #分类结果以字典形式保存，# 使用字典类型储存树的信息
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]            # 复制所有类标签，保证每次递归调用时不改变原始列表的内容
        myTree[bestFeatLabel][value]=createTree(splitDataSet\
                            (dataSet,bestFeat,value),subLabels)
    return myTree

#C4.5
def chooseBestFeatureToSplitByC45(dataSet):
    '''
    选择最好的数据集划分方式
    :param dataSet:
    :return: 划分结果
    '''
    numFeatures = len(dataSet[0]) - 1  # 最后一列yes分类标签，不属于特征变量
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRate = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有维度特征
        infoGainRate = calcInformationGainRatio(dataSet, baseEntropy, i)    # 计算信息增益比 
        if (infoGainRate > bestInfoGainRate):  # 选择最大的信息增益比
            bestInfoGainRate = infoGainRate
            bestFeature = i
    return bestFeature  # 返回最佳特征对应的维度

#执行分类
def classify(inputTree,featLabels,testVec):
    '''
           利用决策树进行分类
    :param: inputTree:构造好的决策树模型
    :param: featLabels:所有的类标签
    :param: testVec:测试数据
    :return: 分类决策结果
    '''
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

#test
if __name__=='__main__':
    # 创造示列数据
    dataSet,labels=createDataSet()
    # 输出决策树模型结果
    print(createTree(dataSet,labels))
    