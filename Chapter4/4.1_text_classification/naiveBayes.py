# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 17:42:11 2018

@author: zhe

E-mail: 1194585271@qq.com
"""
import numpy as np

def loadDataSet(): #import text data
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec 

def createVocabList(dataSet):#create word vector list to contain all text information
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)  #list product
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) #6
    numWords = len(trainMatrix[0]) #32
    pAbusive = sum(trainCategory)/float(numTrainDocs)  #3/6=0.5
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)     #change to ones() #32个1 Laplace smoothing
#    print (p0Num,p1Num)
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0 laplace smoothing
    for i in range(numTrainDocs):#6
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])    #bag of words model (multinomial event model(Andrew Ng))  
#            p1Denom += 1  #set of words model (multi-variate Bernoulli event model(Andrew Ng))
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i]) #bag of words model (multinomial event model(Andrew Ng))  
#            p0Denom += 1 #set of words model (multi-variate Bernoulli event model(Andrew Ng))
#        print (p0Num,p1Num,p0Denom,p1Denom)
    p1Vect = np.log(p1Num/p1Denom)          #change to log()  avoid underflow
    p0Vect = np.log(p0Num/p0Denom)          #change to log()
#    print (p1Vect,p0Vect)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
def testingNBsetOfwords():#convenience function
    listOPosts,listClasses = loadDataSet()      #feature list and label list
    myVocabList = createVocabList(listOPosts)   #generate vocabulary
    trainMat=[]
    for postinDoc in listOPosts:                #genrate feature (0 not exist or 1 exsit) list
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  #differance
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))  #generate tarining parametres 
    
    testEntry = ['love', 'my', 'dalmation','to','dog','part','yes']   #test classficition sample 1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print ("thisdoc featrue vector:",thisDoc)
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage', 'conveninence'] #test classficition sample 2
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print ("thisdoc featrue vector:",thisDoc)
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNBbagofwords():#convenience function
    listOPosts,listClasses = loadDataSet()      #feature list and label list
    myVocabList = createVocabList(listOPosts)   #generate vocabulary
    trainMat=[]
    for postinDoc in listOPosts:                #genrate feature (0 not exist or 1 exsit) list
        trainMat.append(bagOfWords2VecMN(myVocabList, postinDoc))  #differiance
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))  #generate tarining parametres 
    
    testEntry = ['love', 'my', 'dalmation','to','dog','part','yes']   #test classficition sample 1
    thisDoc = np.array(bagOfWords2VecMN(myVocabList, testEntry))
    print ("thisdoc featrue vector:",thisDoc)
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage', 'conveninence','stupid'] #test classficition sample 2
    thisDoc = np.array(bagOfWords2VecMN(myVocabList, testEntry))
    print ("thisdoc featrue vector:",thisDoc)
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))