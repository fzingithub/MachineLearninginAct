# -*- coding: utf-8 -*-
"""
Created on Fri May  4 09:48:38 2018

@author: FZ
"""

import Locally_weighted_linear_regression
import numpy as np
import matplotlib.pyplot as plt

xArr,yArr = Locally_weighted_linear_regression.loadDataSet('ex0.txt') #load data

yHat = Locally_weighted_linear_regression.lwlrTest(xArr,xArr,yArr,0.003) #prediction


#sort
xMat = np.mat(xArr)
srtInd = xMat[:,1].argsort(0)
xSort = xMat[srtInd][:,0,:]


#drawing

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[srtInd])   #fitting line is different when k changes

ax.scatter(xMat[:,1].flatten().A[0],np.mat(yArr).T.flatten().A[0],s=2,c='red') #scatter sample point 

plt.savefig('lwlr0_003.eps',dpi=2000)
plt.show()


 