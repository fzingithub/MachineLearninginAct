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

#计算w
Mw = np.matrix(np.zeros(np.shape(dataArr)[1]))
for i in range (np.shape(alpha)[0]):
	if alpha[i]>0:
		Mw += np.multiply(labelArr[i]*alpha[i],dataArr[i])
		
w = Mw.T.tolist()
print (w)


#drawing
n = np.shape(labelArr)[0] 
xcord1 = []; ycord1 = []   
xcord2 = []; ycord2 = []
xcord3 = []; ycord3 = []
xcord4 = []; ycord4 = []
for i in range(n):
	if int(labelArr[i])== 1:
		if alpha[i]>0:
			xcord3.append(dataArr[i][0]); ycord3.append(dataArr[i][1])
		else:
			xcord1.append(dataArr[i][0]); ycord1.append(dataArr[i][1])
	else:
		if alpha[i]>0:
			xcord4.append(dataArr[i][0]); ycord4.append(dataArr[i][1])
		else:
			xcord2.append(dataArr[i][0]); ycord2.append(dataArr[i][1]) 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord1, ycord1, s=40, c='yellow', marker='s')
ax.scatter(xcord2, ycord2, s=40, c='green')
ax.scatter(xcord3, ycord3, s=50, c='red',marker='s')		
ax.scatter(xcord4, ycord4, s=50, c='red')	

x = np.arange(2.8, 6.5, 0.1)
y1 = (b+(w[0][0])*x)/(-w[1][0])
y = np.mat(y1).T
ax.plot(x, y)
plt.xlabel('X1'); plt.ylabel('X2');
plt.savefig('SVMSMO.eps',dpi=2000)
plt.show()