# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:48:03 2018

@author: FZ
"""

import numpy as np


# =============================================================================
# process data
# =============================================================================
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 2
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t') #['1.000000', '0.116163', '3.129283']
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)   #[[1.0, 0.52707], [1.0, 0.116163],...]
        labelMat.append(float(curLine[-1])) #[4.225236, 4.231083,...]
    return dataMat,labelMat  #list

# =============================================================================
# linear regression
# =============================================================================


def standRegres(xArr,yArr):
    xMat = np.mat(xArr) 
    yMat = np.mat(yArr).T 
    xTx = xMat.T*xMat  #xMat.T*xMat*w - xMat.T*yMat = 0
    if np.linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


# =============================================================================
# locally weighted linear regression
# =============================================================================
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]   #difference matrix
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))   #weighted matrix
    xTx = xMat.T * (weights * xMat)      
    if np.linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))   #normal equation
    #ws #7 feature,and 1 
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = np.shape(testArr)[0]   # #trianing sample
    yHat = np.zeros(m) #array
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


#others   
def rssError(yArr,yHatArr):
      return ((yArr-yHatArr)**2).sum()