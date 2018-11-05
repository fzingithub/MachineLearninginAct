# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 13:43:40 2018

@author: zhe

E-mail: 1194585271@qq.com
"""

import trees
import treePlotter

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]

lenseLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = trees.createTree(lenses,lenseLabels)

treePlotter.createPlot(lensesTree)
