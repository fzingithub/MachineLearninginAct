# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 20:27:04 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

import naiveBayes

listOPosts , listClasses = naiveBayes.loadDataSet()
myVocabList = naiveBayes.createVocabList(listOPosts)

print ("文档模型总词表：",myVocabList) #vocabulary

wordVec = naiveBayes.setOfWords2Vec(myVocabList,listOPosts[0])

print ("测试文档词向量：",wordVec)

trainMat = []

for postinDoc in listOPosts:
    trainMat.append(naiveBayes.setOfWords2Vec(myVocabList,postinDoc))
    
p0V,p1V,PAb = naiveBayes.trainNB0(trainMat,listClasses)

print ("词表元素对0类的概率贡献值向量:",p0V)
print ("词表元素对1类的概率贡献值向量:",p1V)
print ("文档为0类概率",1-PAb)
print ("文档为1类概率",PAb)


    
