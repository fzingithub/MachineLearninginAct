# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 09:05:34 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

import numpy as np
import operator 
from os import listdir
import time 
import kdtree

# This class emulates a tuple, but contains a useful payload
class Item(object):
    def __init__(self, array, data):
        self.coords = array
        self.data = data

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        return self.coords[i]

    def __repr__(self):
        return 'Item({}, {}, {})'.format(self.coords[0], self.coords[1], self.data)

def createkdtree(dataSet, labels):
    listtree = []
    for i in range(len(dataSet)):
        listtree.append(Item(dataSet[i],labels[i]))
    tree = kdtree.create(listtree)
    return tree

def classifykdtree(inX, tree, k):
    resultnodelist = tree.search_knn(inX,k)
    dictresult = {i:0 for i in range(10)}
    for i in range(k):
        dictresult[resultnodelist[i][0].data.data] += 1
    sortedClassCount = sorted(dictresult.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



def classify0(inX, dataSet, labels, k):
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
    return sortedClassCount[0][0]

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    
    tree = createkdtree(trainingMat,hwLabels)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classifykdtree(vectorUnderTest[0,:], tree, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))
    
if __name__=="__main__":
    start=time.clock()
    handwritingClassTest()
    end=time.clock()
    total_time=end-start
    print("Time For Run KNN:"+str(total_time))
    
#the total number of errors is: 10
#the total error rate is: 0.010571
#Time For Run CompleteSMOWithKernel:34.74240867341453s compared with SVM 9s