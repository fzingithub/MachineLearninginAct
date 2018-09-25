# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:18:32 2018

@author: zhe

E-mail：1194585271@qq.com
"""

import svmMLiA
import numpy as np

start=time.clock()

kTup=('rbf', 10)
dataArr,labelArr = svmMLiA.loadImages('trainingDigits')
b,alphas = svmMLiA.smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
svInd=np.nonzero(alphas.A>0)[0]
sVs=datMat[svInd] 
labelSV = labelMat[svInd];
print ("there are %d Support Vectors" % np.shape(sVs)[0])
m,n = np.shape(datMat)
errorCount = 0
for i in range(m):
	kernelEval = svmMLiA.kernelTrans(sVs,datMat[i,:],kTup)
	predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
	if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1
print ("the training error rate is: %f" % (float(errorCount)/m))
dataArr,labelArr = svmMLiA.loadImages('testDigits')
errorCount = 0
datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
m,n = np.shape(datMat)
for i in range(m):
	kernelEval = svmMLiA.kernelTrans(sVs,datMat[i,:],kTup)
	predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
	if np.sign(predict)!=np.sign(labelArr[i]): errorCount += 1    
print ("the test error rate is: %f" % (float(errorCount)/m) )

end=time.clock()
total_time=end-start
print("Time For Run CompleteSMOWithKernel:"+str(total_time))