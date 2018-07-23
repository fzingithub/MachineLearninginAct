# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 20:27:04 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

import naiveBayes

listOPosts , listClasses = naiveBayes.loadDataSet()
myVocabList = naiveBayes.createVocabList(listOPosts)

print (myVocabList) #vocabulary

wordVec = naiveBayes.setOfWords2Vec(myVocabList,listOPosts[0])

print (wordVec)

trainMat = []

for postinDoc in listOPosts:
    trainMat.append(naiveBayes.setOfWords2Vec(myVocabList,postinDoc))
    
p0V,p1V,PAb = naiveBayes.trainNB0(trainMat,listClasses)


    
