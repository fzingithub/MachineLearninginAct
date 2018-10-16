# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:18:04 2018

@author: zhe

E-mail：1194585271@qq.com
"""

import svmMLiA
import numpy as np
import time

start=time.clock()

#k1 = 1.2
#dataArr,labelArr = svmMLiA.loadDataSet('testSetRBF.txt')
#datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
#b,alphas = svmMLiA.smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
#svInd=np.nonzero(alphas.A>0)[0]
#sVs=datMat[svInd] #get matrix of only support vectors；
#labelSV = labelMat[svInd];
#print ("there are %d Support Vectors" % np.shape(sVs)[0])
#m,n = np.shape(datMat)
#errorCount = 0
#for i in range(m):
#	kernelEval = svmMLiA.kernelTrans(sVs,datMat[i,:],('rbf', k1))
#	predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
#	if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
#print ("the training error rate is: %f" % (float(errorCount)/m))
#dataArr,labelArr = svmMLiA.loadDataSet('testSetRBF2.txt')
#errorCount = 0
#datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
#m,n = np.shape(datMat)
#for i in range(m):
#	kernelEval = svmMLiA.kernelTrans(sVs,datMat[i,:],('rbf', k1))
#	predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
#	if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1    
#print ("the test error rate is: %f" % (float(errorCount)/m))
svmMLiA.testRbf()

end=time.clock()
total_time=end-start
print("Time For Run CompleteSMOWithKernel:"+str(total_time))