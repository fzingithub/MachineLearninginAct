# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:59:16 2018

@author: FZ
"""
import matplotlib.pyplot as plt
import numpy as np
import linear_regression

xArr,yArr = linear_regression.loadDataSet('ex0.txt') #得到特征矩阵 以及对应的目标矩阵

ws = linear_regression.standRegres(xArr,yArr) #得到回归系数

#ouput : y = ws[0] + ws[1]*x
#drawing

xMat = np.mat(xArr)
yMat = np.mat(yArr)
yHat = xMat*ws      #prediction value

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],color='k')

xCopy = xMat.copy()
xCopy.sort(0) #升序防止次序混乱
yCHat = xCopy*ws

ax.plot(xCopy[:,1],yCHat,color='k')

plt.savefig('linear_fitting.eps',dpi=2000)
plt.show()

#判断拟合好坏，算相关系数

correlation_coefficient = np.corrcoef(yHat.T,yMat)
#1	0.986474
#0.986474	1 模型拟合度 好