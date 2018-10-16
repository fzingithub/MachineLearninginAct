# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 20:27:04 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

import naiveBayes

listOPosts , listClasses = naiveBayes.loadDataSet()
myVocabList = naiveBayes.createVocabList(listOPosts)

print ("Vocubulary：",myVocabList) #vocabulary

wordVec = naiveBayes.setOfWords2Vec(myVocabList,listOPosts[0])

print ("test word vector：",wordVec)

trainMat = []

for postinDoc in listOPosts:
    trainMat.append(naiveBayes.setOfWords2Vec(myVocabList,postinDoc))
    
p0V,p1V,PAb = naiveBayes.trainNB0(trainMat,listClasses)

print ("Probability vector for 0 classification:",p0V)
print ("Probability vector for 1 classification:",p1V)
print ("Probability of being 0 classification:",1-PAb)
print ("Probability of being 1 classification",PAb)

print ("set of words models:=====================================================")
naiveBayes.testingNBsetOfwords()
print ("=========================================================================")
print ("\nbag of words models:=====================================================")
naiveBayes.testingNBbagofwords()
print ("=========================================================================")


    
