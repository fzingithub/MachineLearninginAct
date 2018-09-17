# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:01:57 2018

@author: zhe

E-mail：1194585271@qq.com
"""

import  numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')     #delete ' ',\n  split with '\t'
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):  #s.t. 0<=\alpha<=C
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn); labelMat = np.mat(classLabels).transpose()
    b = 0; m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m);print("i",i,alphas[i]);print("j",j,alphas[j]);                                                                          #adddddddddddddddddddddddddddddddddddddddddddd
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print ("L==H"); continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print ("eta>=0"); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print ("j not moving enough",alphas[j]);continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j]);print("i",i,alphas[i]);print("j",j,alphas[j]);#update i by the same amount as j   #adddddddddddddddddddddddddddddddddddddddddd
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print ("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print ("iteration number: %d" % iter)
    return b,alphas


#by myself
def calculateW(dataArr,labelArr,alpha,b):
	Mw = np.matrix(np.zeros(np.shape(dataArr)[1]))
	for i in range (np.shape(alpha)[0]):
		if alpha[i]>0:
			Mw += np.multiply(labelArr[i]*alpha[i],dataArr[i])
	w = Mw.T.tolist()
	return w

#by myself
def drawing(dataArr,labelArr,alpha,b):
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
	ax.scatter(xcord1, ycord1, s=40, c='yellow', marker='s',label='class 1')
	ax.scatter(xcord2, ycord2, s=40, c='green',label='class -1')
	ax.scatter(xcord3, ycord3, s=40, c='red',marker='s',label='SV')		
	ax.scatter(xcord4, ycord4, s=40, c='red',label='SV')	

	ax.legend(loc='best')

	x = np.arange(2.7, 6.6, 0.1)
	y1 = (b+(calculateW(dataArr,labelArr,alpha,b)[0][0])*x)/(-calculateW(dataArr,labelArr,alpha,b)[1][0])
	y = np.mat(y1).T
	ax.plot(x, y,'-')
	plt.xlabel('X1'); plt.ylabel('X2');
	plt.savefig('SMOSimple.eps',dpi=2000)
	plt.show()



