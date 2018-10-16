from numpy import *
import logRegres_stocGradAscent

#stocGradAscent
dataArr,labelMat = logRegres_stocGradAscent.loadDataSet()
weights = logRegres_stocGradAscent.stocGradAscent0(array(dataArr), labelMat)
logRegres_stocGradAscent.plotBestFit(weights)

#pro-stocGradAscent
dataArr,labelMat = logRegres_stocGradAscent.loadDataSet()
weights = logRegres_stocGradAscent.stocGradAscent1(array(dataArr), labelMat,50)
logRegres_stocGradAscent.plotBestFit(weights)
