# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 18:53:24 2018

@author: zhe

E-mail：1194585271@qq.com
"""

import svmMLiA
import time
import matplotlib.pyplot as plt
import  numpy as np

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
print("总耗时:"+str(total_time))

#drawing

x = list(np.mat(dataArr).T[0])
y = list(np.mat(dataArr).T[1])
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
#设置标题  
ax1.set_title('Scatter Plot')  
#设置X轴标签  
plt.xlabel('X')  
#设置Y轴标签  
plt.ylabel('Y')  
#画散点图  
ax1.scatter(x,y,c = 'r',marker = 'o')  
#设置图标  
plt.legend('x1')  
#显示所画的图  
plt.show()  