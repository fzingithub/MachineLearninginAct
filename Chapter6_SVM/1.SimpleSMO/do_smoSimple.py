# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 18:53:24 2018

@author: zhe

E-mail：1194585271@qq.com
"""

import svmMLiA
import time

start=time.clock()

dataArr,labelArr = svmMLiA.loadDataSet('testSet.txt')

b,alpha = svmMLiA.smoSimple(dataArr,labelArr,0.6,0.001,80)

print (b)

print (alpha[alpha>0])

for i in range(100):
	if alpha[i]>0.0:
		print (i,dataArr[i],labelArr[i])
		


end=time.clock()
total_time=end-start
print("Time For Run SMO:"+str(total_time))

w = svmMLiA.calculateW(dataArr,labelArr,alpha)
#drawing
svmMLiA.drawing(dataArr,labelArr,alpha,w,b)



