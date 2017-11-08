#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import operator
import numpy as np
import matplotlib.pyplot as plt

def file2matrix(file_path):
  """
  desc:  
    导入训练数据集，转化为特征矩阵和标签向量  
  params:  
    file_path: 数据集路径，相对于程序运行目录，字符串  
  return:  
    attr_mat：属性矩阵，np 对象  
    label_vec：数据集类别，原生数组  
  """
  print('file path -->', file_path)

  # 确定文件行数，预先指定返回结果
  fd = open(file_path)
  line_len = len(fd.readlines())
  attr_mat = np.zeros((line_len, 3))
  label_vec = []

  fd = open(file_path)
  index = 0
  for line in fd.readlines():
    line = line.strip()
    data = line.split('\t')

    # 
    # 技巧
    # array[,]中,表示维度，或者行
    # array[:]中:表示区间，或者列
    # array[1, :]第一行的所有列数据
    # array[:, 1]所有行的第一列数据
    # 
    attr_mat[index, :] = data[0:3]
    label_vec.append(int(data[-1]))
    index += 1
  return attr_mat, label_vec

def draw_scatter(datax, datay, datal, title = 'scatter title', xlabel = 'scatter xlabel', ylabel = 'scatter ylabel'):
  """
  desc:  
    根据数据集，分类向量画图  
  params:  
    datax: x 轴数据集，np 数组对象  
    datay: y 轴数据集，np 数组对象  
    datal: 数据集分类标签，原生数组  
    title: 标题，字符串  
    xlabel: x 轴标签，字符串  
    ylabel: y 轴标签，字符串  
  return:  
    None
  """

  # 配置标题，标签，刻度
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.xlim(0, datax.max(0))
  plt.ylim(0, datay.max(0))

  # 配置大小，颜色，样式，透明度，边线宽，边线颜色
  sizes = []
  colors = []
  marker = 'o'
  alpha = 0.5
  linewidths = []
  edgecolors = []
  for temp in datal:
    sizes.append(30)    
    linewidths.append(1)
    if temp == 1:
      colors.append('#4AB8EA')
    elif temp == 2:
      colors.append('#E2769F')
    elif temp == 3:
      colors.append('#F5C224')
    else:
      colors.append('#E94F5A')
      edgecolors.append('#F18754')

  plt.scatter(datax, datay, s = sizes, c = colors, marker = marker, alpha = alpha, linewidths = linewidths, edgecolors = edgecolors)
  plt.show()

def auto_normal(data):
  """
  desc:  
    归一化数据集  
    公式：y = (x - min_x)/(max_x - min_x)  
  params:  
    data: 数据集，np 数组对象   
  return:  
    norm_data: 归一化数据集，np 数组对象   
    min_data: 最小值数据集，np 数组对象   
    range_data: 极差数据集，np 数组对象   
  """

  # 
  # 技巧
  # np.max(array, axis), np.min(array, axis)
  # axis缺失，所有元素一起比较
  # axis=0，列元素一起比较
  # axis=1，行元素一起比较
  # 
  min_data = data.min(0)
  max_data = data.max(0)
  range_data = max_data - min_data
  norm_data = np.zeros(data.shape)
  m = data.shape[0]
  norm_data = data - np.tile(min_data, (m, 1))
  norm_data = norm_data / np.tile(range_data, (m, 1))
  return norm_data, min_data, range_data

def classify_knn(attr_vec, attr_mat, label_vec, k):
  """
  desc:  
    knn 预测分类标签  
  params:  
    attr_vec: 待预测属性向量，原生数组  
    attr_mat: 属性矩阵，np 对象  
    label_vec: 数据集类别，原生数组   
    k: 最近邻个数，数值   
  return:  
    label: 分类标签，字符串   
  """

  m = attr_mat.shape[0]
  diff_mat = np.tile(attr_vec, (m, 1)) - attr_mat
  sqdiff_mat = diff_mat ** 2
  sq_distances = sqdiff_mat.sum(axis = 1)
  distances = sq_distances ** 0.5
  dis_indexes = distances.argsort()
  label_dict = {}
  for index in range(k):
    temp = label_vec[dis_indexes[index]]
    label_dict[temp] = label_dict.get(temp, 0) + 1
  label_sort = sorted(label_dict.items(), key = operator.itemgetter(1), reverse=True)
  label = label_sort[0][0]
  return label

def clean_data(file_path):
  """
  desc:  
    清洗并展示数据  
  params:  
    file_path: 数据集文件路径 
  return:  
    norm_data: 归一化数据集，np 数组对象   
    min_data: 最小值数据集，np 数组对象   
    range_data: 极差数据集，np 数组对象  
  """

  attr_mat, label_vec = file2matrix(file_path)
  # print(attr_mat, label_vec)
  print('展示原始数据')
  draw_scatter(attr_mat[:, 0], attr_mat[:, 1], label_vec, 'dating', 'flying', 'video')
  draw_scatter(attr_mat[:, 0], attr_mat[:, 2], label_vec, 'dating', 'flying', 'eating')
  draw_scatter(attr_mat[:, 1], attr_mat[:, 2], label_vec, 'dating', 'video', 'eating')

  norm_data, min_data, range_data = auto_normal(attr_mat)
  print('展示归一化数据')
  draw_scatter(norm_data[:, 0], norm_data[:, 1], label_vec, 'dating', 'flying', 'video')
  draw_scatter(norm_data[:, 0], norm_data[:, 2], label_vec, 'dating', 'flying', 'eating')
  draw_scatter(norm_data[:, 1], norm_data[:, 2], label_vec, 'dating', 'video', 'eating')

  return norm_data, min_data, range_data

def classify_test(file_path, data_ratio):
  """
  desc:  
    测试 knn 分类效果  
  params:  
    data_ratio: 测试数据占比，数值  
  return:  
    classify_ratio: knn 分类错误占比，数值   
  """

  attr_mat, label_vec = file2matrix(file_path)
  norm_data, min_data, range_data = auto_normal(attr_mat)
  m = norm_data.shape[0]
  test_len = int(m * data_ratio)
  error_count = 0.0
  for index in range(test_len):
    label = classify_knn(norm_data[index, :], norm_data[test_len:m, :], label_vec[test_len:m], 3)
    print('knn classify predict %d, actual is %d' % (label, label_vec[index]))
    if (label != label_vec[index]):
      error_count += 1.0
  classify_ratio = error_count/float(test_len)
  print('knn classify error ratio is %f' % classify_ratio)
  return classify_ratio

def study():
  mat = [[1, 2, 3], [9, 8, 6], [7, 0, 4]]
  vec = [1, 2, 3]
  mat = np.array(mat)
  m = mat.shape
  print('矩阵 行列')
  print(m)
  max_mat = mat.max(axis=0)
  print('矩阵 max')
  print(max_mat)
  tile_mat = np.tile(vec, (m[0], 1))
  print('矩阵 tile')
  print(tile_mat)
  diff_mat = tile_mat - mat
  print('矩阵 差值')
  print(diff_mat)
  sq_mat = diff_mat ** 2
  print('矩阵 平方')
  print(sq_mat)
  distance = sq_mat.sum(axis = 1)
  print('矩阵 和')
  print(distance)
  dis_mat = distance ** 0.5
  print('矩阵 开方')
  print(dis_mat)
  dis_index = distance.argsort()
  print('矩阵 排序索引')
  print(dis_index)

  name_dict = {
    'name': 'name dict'
  }
  print('dict get function')
  print(name_dict)
  print(name_dict.get('name'))
  print(name_dict.get('age', 'age no init'))

  num_dict = {
    'a': 1,
    'c': 3,
    'd': 2
  }
  print('sorted function')
  print(num_dict)
  print(num_dict.items())
  print(sorted(num_dict.items(), key = operator.itemgetter(1), reverse=False))


if __name__ == '__main__':
  print('start knn...')

  file_path = 'data/dating.txt'

  # clean_data(file_path)

  # classify_test(file_path, 0.1)

  print('knn over...')

  study()
