# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:59:41 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

import numpy as np

def createVocabList(dataSet):#create word vector list to contain all text information
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) 
    numWords = len(trainMatrix[0]) 
    pAbusive = (sum(trainCategory)+1)/(float(numTrainDocs)+2)      #laplace smoothing
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)     #change to ones() #32个1 Laplace smoothing
#    print (p0Num,p1Num)
    p0Denom = numWords; p1Denom = numWords                        #change to #vocabulary laplace smoothing
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

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

def textParse(bigString):    #input is big string, #output is word list
    import re
    pattern = re.compile('\\W+')
    listOfTokens = pattern.split(bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  #return list
    
def spamTest():#process function
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)        #list of list
        fullText.extend(wordList)       #list of word
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)        #create vocabulary
    trainingSet = list(range(50)); testSet=[]           #create test set
    
    print ("all valid words number:",len(fullText))
    print ("The length of word vector:",len(vocabList))
    
    for i in range(10):#select randomly 10 mails as test set
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
        
    trainMat=[]; trainClasses = []   #train naive bayes classifier in trainSet 
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    
    
    errorCount = 0          #test classifier
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])   #numpy array
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print ("classification error",docIndex,docList[docIndex])
    print ('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText