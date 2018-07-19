# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:53:58 2018

@author: FZ
"""

import function
import numpy as np

abX,abY = function.loadDataSet('abalone.txt')

#locally weighted linear regression
yHat01 = function.lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1) #training set #0-99
yHat1 = function.lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
yHat10 = function.lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)

#error
error01 = function.rssError(abY[0:99],yHat01.T)   #56.820227823572182
error1 = function.rssError(abY[0:99],yHat1.T)   #429.89056187016683
error10 = function.rssError(abY[0:99],yHat10.T)   #549.1181708825128


#generalization 
yHat01g = function.lwlrTest(abX[100:199],abX[100:199],abY[100:199],0.1) #test set #100-199
yHat1g = function.lwlrTest(abX[100:199],abX[100:199],abY[100:199],1)
yHat10g = function.lwlrTest(abX[100:199],abX[100:199],abY[100:199],10)

#error
error01g = function.rssError(abY[100:199],yHat01g.T)   #36199.797699875046
error1g = function.rssError(abY[100:199],yHat1g.T)   #231.81344796874004
error10g = function.rssError(abY[100:199],yHat10g.T)   #291.87996390562728

#compare
#k = 0.1 overfitting


#linear regression
ws = function.standRegres(abX[0:99],abY[0:99])
yHat = np.mat(abX[100:199])*ws
errorlr = function.rssError(yHat.T.A,abY[100:199])   #518.63631532510897

#compare
#lwlr is better than lr