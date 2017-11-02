#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import operator
import numpy as np

# inX 待预测数据对象，list对象
# dataSet 训练集特征，numpy对象
# labels 训练集标签，list对象
# k k近邻，数值
# 返回类别
def classify(inX, dataSet, labels, k):
  dataSetSize = dataSet.shape[0]
  diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
  sqDiffMat = diffMat**2
  sqDistances = sqDiffMat.sum(axis=1)
  distances = sqDistances**0.5
  sortedDistIndicies = distances.argsort()
  classCount = {}
  for i in range(k):
    voteIlabel = labels[sortedDistIndicies[i]]
    classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
  sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
  return sortedClassCount[0][0]

# dataSet 待归一化数组，numpy对象
# 返回二维数组，numpy对象
def autoNorm(dataSet):
  minVal = dataSet.min(0)
  maxVal = dataSet.max(0)
  rangeVal = maxVal - minVal
  normDataSet = np.zeros(np.shape(dataSet))
  m = dataSet.shape[0]
  normDataSet = dataSet - np.tile(minVal, (m, 1))
  normDataSet = normDataSet/np.tile(rangeVal, (m, 1))
  return normDataSet, rangeVal, minVal

def create_data():
  group = np.array([[100, 10], [90, 9], [90, 10], [100, 90], [80, 8], [0, 0], [10, 1], [0, 10], [10, 0], [20, 2]])
  labels = ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
  return group, labels

def main():
  group, labels = create_data()
  test = [20, 2]
  label = classify(test, group, labels, 3)
  print(label)

if __name__ == '__main__':
  main()