# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:19:13 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        for i in range(len(curLine)):
            curLine[i] = float(curLine[i])
        dataMat.append(curLine)
    return dataMat




if __name__=='__main__':
    
    plt.figure(figsize=(10, 10))
    
    datMat = np.array(loadDataSet('testSet.txt'))
    
    y_pred = KMeans(n_clusters=4, random_state=0).fit_predict(datMat)
    
    plt.subplot(221)
    plt.scatter(datMat[:, 0], datMat[:, 1], c = y_pred)
    plt.title("Kmeans result1")
    
    datMat2 = np.array(loadDataSet('testSet2.txt'))
    
    y_pred2 = KMeans(n_clusters=3, random_state=0).fit_predict(datMat2)
    
    
    plt.subplot(222)
    plt.scatter(datMat2[:, 0], datMat2[:, 1], c = y_pred2)
    plt.title("Kmeans result2")
    plt.savefig('KMeans.eps',dpi=1000)