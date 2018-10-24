# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:59:00 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

import KNN
import numpy as np
import operator

dataSet,labels = KNN.createDataSet()

k=3
inX = 4
dataSetSize = dataSet.shape[0]   #training sample
diffMat = np.tile(inX, (dataSetSize,1)) - dataSet   #difference
sqDiffMat = diffMat**2   #square
sqDistances = sqDiffMat.sum(axis=1)   #axis=1 row; 0 column
distances = sqDistances**0.5   #root
sortedDistIndicies = distances.argsort()   #sort to get index       
classCount={}          
for i in range(k):
    voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  # value default 0
sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #iterator,Specified sort field,Descending order

print (sortedClassCount)
