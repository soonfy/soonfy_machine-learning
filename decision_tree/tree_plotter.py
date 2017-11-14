#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@module plot tree
@methods
  1. get_leaf
  2. get_depth
  3. createPlot

@desc study plot decision tree
@author soonfy<soonfy@163.com>
@log
created at: 2017-11-10
"""
print(__doc__)

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = 'sawtooth', fc = '0.8')
leafNode = dict(boxstyle = 'round4', fc = '0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeTxt, cntrPt, parentPt, nodeType):
  createPlot.ax1.annotate(nodeTxt, xy = parentPt, xycoords = 'axes fraction', xytext = cntrPt, textcoords = 'axes fraction', 
  va = 'center', ha = 'center', bbox = nodeType, arrowprops = arrow_args)

def get_leaf(tree):
  """
  desc:  
    返回树状字典的节点数目 
  params:  
    tree: 树状字典，原生字典   
  return:  
    leaf: 节点数目   
  """

  leaf = 0

  # 
  # dict.keys()在python2.x中返回一个列表，在python3.x中返回一个dict_keys对象
  # 对象转数组：list(dict.keys())[index]
  # 
  root = list(tree.keys())[0]
  super_leaf = tree[root]
  for sub_leaf in super_leaf.keys():
    if type(super_leaf[sub_leaf]).__name__ == 'dict':
      leaf += get_leaf(super_leaf[sub_leaf])
    else:
      leaf += 1
  return leaf

def get_depth(tree):
  """
  desc:  
    返回树状字典的深度 
  params:  
    tree: 树状字典，原生字典   
  return:  
    depth: 深度   
  """

  depth = 0
  root = list(tree.keys())[0]
  super_leaf = tree[root]
  for sub_leaf in super_leaf.keys():
    if type(super_leaf[sub_leaf]).__name__ == 'dict':
      this_depth = 1 + get_depth(super_leaf[sub_leaf])
    else:
      this_depth = 1
    if this_depth > depth:
      depth = this_depth
  return depth

def plotMideText(cntrPt, parentPt, txtString):
  xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
  yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
  createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
  leaf = get_leaf(myTree)
  depth = get_depth(myTree)
  firstStr = list(myTree.keys())[0]
  cntrPt = (plotTree.xOff + (1.0 + float(leaf)) / 2.0 / plotTree.totalW, plotTree.yOff)
  plotMideText(cntrPt, parentPt, nodeTxt)
  plotNode(firstStr, cntrPt, parentPt, decisionNode)
  secondDict = myTree[firstStr]
  plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
  for key in secondDict.keys():
    if type(secondDict[key]).__name__ == 'dict':
      plotTree(secondDict[key], cntrPt, str(key))
    else:
      plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
      plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
      plotMideText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
  plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(tree):
  """
  desc:  
    根据决策树的嵌套字典画图  
  params:  
    tree: 表示决策树的嵌套字典   
  return:  
    None
  """

  fig = plt.figure(1, facecolor = 'white')
  fig.clf()
  axprops = dict(xticks = [], yticks = [])
  createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
  plotTree.totalW = float(get_leaf(tree))
  plotTree.totalD = float(get_depth(tree))
  plotTree.xOff = -0.5 /plotTree.totalW; plotTree.yOff = 1.0
  plotTree(tree, (0.5, 1.0), '')
  plt.show()
