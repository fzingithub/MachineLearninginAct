# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:19:49 2018

@author: zhe

E-mail：1194585271@qq.com
"""

import svmMLiA
import time

start=time.clock()

dataArr,labelArr = svmMLiA.loadDataSet('testSet.txt')

b,alpha = svmMLiA.smoP(dataArr,labelArr,0.6,0.001,40)

print (b)

print (alpha[alpha>0.00])

for i in range(100):
	if alpha[i]>0.00:
		print (i,dataArr[i],labelArr[i])
		


end=time.clock()
total_time=end-start
print("Time For Run CompleteSMO:"+str(total_time))

#drawing
w = svmMLiA.calculateW(dataArr,labelArr,alpha)

#drawing
svmMLiA.drawing(dataArr,labelArr,alpha,w,b)