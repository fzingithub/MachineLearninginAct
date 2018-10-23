# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:59:00 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

import KNN

dataSet,labels = KNN.createDataSet()

result = KNN.classify0([0,0],dataSet,labels,3)
