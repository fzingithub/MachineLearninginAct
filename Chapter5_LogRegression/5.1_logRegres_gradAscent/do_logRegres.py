'''
Created on June 9,2018
do logregers_gradAscent
@author: zhe
'''

import logRegres
from numpy import *

dataArr,labelMat = logRegres.loadDataSet()

logRegres.gradAscent(dataArr,labelMat)

weights = logRegres.gradAscent(dataArr,labelMat)

logRegres.plotBestFit(weights.getA())