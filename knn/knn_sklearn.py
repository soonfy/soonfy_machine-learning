#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@module knn_sklearn
@methods
  1. classify_knn_sk

@desc study scikit-learn neighbors
@author soonfy<soonfy@163.com>
@log
created at: 2017-11-09
"""
print(__doc__)

import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt
from matplotlib import colors

import knn as knn

def classify_knn_sk(feature_mat, label_vec, n_neighbors):
  """
  desc:  
    scikit-learn neighbors 预测分类标签  
  params:  
    feature_mat: 特征属性矩阵，必须进行归一化，np 数组对象  
    label_vec: 数据集类别，原生数组   
    n_neighbors: 最近邻个数，数值   
  return:  
    None
  """

  colormesh = colors.ListedColormap(['#4AB8EA', '#E2769F', '#F5C224'])
  colorscatter = colors.ListedColormap(['#0678EA', '#A0269F', '#B08224'])

  step = 0.1
  for weights in ['uniform', 'distance']:
    neigh = neighbors.KNeighborsClassifier(n_neighbors, weights)
    neigh.fit(feature_mat, label_vec)
    x_min, x_max = feature_mat[:, 0].min(), feature_mat[:, 0].max()
    y_min, y_max = feature_mat[:, 1].min(), feature_mat[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    labels = neigh.predict(np.c_[xx.ravel(), yy.ravel()])

    labels = labels.reshape(xx.shape)
    plt.figure()
    plt.title('scikit-learn neighbors(k = %d, weights = "%s")' % (n_neighbors, weights))
    plt.pcolormesh(xx, yy, labels, cmap = colormesh)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.scatter(feature_mat[:, 0], feature_mat[:, 1], c = label_vec, cmap = colorscatter)
  plt.show()

def study():
  x_ran = np.arange(1, 10, 2)
  y_ran = np.arange(1, 10, 3)
  print(x_ran)
  print(y_ran)
  xx, yy = np.meshgrid(x_ran, y_ran)
  print(xx)
  print(yy)
  print(xx.ravel())
  print(yy.ravel())
  print(np.c_[xx.ravel(), yy.ravel()])

if __name__ == '__main__':
  print('start scikit-learn neighbors...')
  file_path = 'data/dating.txt'
  feature_mat, label_vec = knn.file2matrix(file_path)
  feature_mat = feature_mat[:, 0 : 2]
  norm_data, min_data, range_data = knn.auto_normal(feature_mat)
  classify_knn_sk(norm_data, label_vec, 3)
  print('scikit-learn neighbors over...')

  study()
