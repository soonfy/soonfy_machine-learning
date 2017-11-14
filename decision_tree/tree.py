#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@module tree
@methods
  1. file2matrix
  2. calculate_shannon_entropy
  3. split_data
  4. choose_max_feature
  5. create_tree
  5. store_tree
  5. grab_tree

@desc study decision tree
@author soonfy<soonfy@163.com>
@log
created at: 2017-11-10
"""
print(__doc__)

import math
import operator
import tree_plotter as tree_plotter

def file2matrix(file_path):
  """
  desc:  
    导入训练数据集，转化为特征属性数组和特征属性集合  
  params:  
    file_path: 数据集路径，相对于程序运行目录，字符串  
  return:  
    feature_list：特征属性数组，原生数组对象  
    labels：特征属性集合，原生数组  
  """
  print('file path -->', file_path)

  # 确定文件行数，预先指定返回结果
  fd = open(file_path)
  feature_list = [line.strip().split('\t') for line in fd.readlines()]
  labels = ['age', 'prescript', 'astigmatic', 'tearRate']
  return feature_list, labels

def calculate_shannon_entropy(data_set):
  """
  desc:  
    计算香农熵  
    公式：y = -sum[1-n](p(x) * log(p(x), 2))
    参数：x是分类值，y是香农熵，sum[1-n]求累加，p(x)是x类别的概率，log(p(x), 2)是以2为底，p(x)的对数    
  params:  
    data_set: 数据集，原生数组对象   
  return:  
    shannon_entropy: 数据集的香农熵，浮点数值   
  """

  shannon_entropy = 0.0
  data_len = len(data_set)
  label_count = {}
  for data in data_set:
    label = data[-1]
    label_count[label] = label_count.get(label, 0) + 1
  for label in label_count:
    prob = float(label_count[label])/data_len
    shannon_entropy -= prob * math.log(prob, 2)
  return shannon_entropy

def split_data(data_set, axis, value):
  """
  desc:  
    根据特征值划分子数据集  
  params:  
    data_set: 数据集，原生数组对象   
    axis: 特征位置，数值   
    value: 特征值   
  return:  
    sub_data_set: 根据特征值划分出的子数据集   
  """

  sub_data_set = []
  for data in data_set:
    if data[axis] == value:
      temp = data[:axis]
      temp.extend(data[axis + 1:])
      sub_data_set.append(temp)
  return sub_data_set

def choose_max_feature(data_set):
  """
  desc:  
    根据信息增益找出最优划分属性  
  params:  
    data_set: 数据集，原生数组对象   
  return:  
    max_feature: 信息增益最高的特征属性索引   
    # max_gain: 最高信息增益   
    # feature_gain: 特征属性对应的信息增益   
  """

  feature_len = len(data_set[0]) - 1
  set_entropy = calculate_shannon_entropy(data_set)
  max_gain = 0.0; max_feature = -1
  feature_gain = {}
  for index in range(feature_len):
    feature_values = [data[index] for data in data_set]
    feature_values = set(feature_values)
    data_entropy = 0.0
    for feature_value in feature_values:
      sub_data = split_data(data_set, index, feature_value)
      prob = len(sub_data)/float(len(data_set))
      data_entropy += prob * calculate_shannon_entropy(sub_data)
    gain = set_entropy - data_entropy
    feature_gain[index] = gain
    if gain > max_gain:
      max_gain = gain
      max_feature = index
  # return max_feature, max_gain, feature_gain
  return max_feature

def choose_max_label(labels):
  """
  desc:  
    根据分类标签出现的频率确定节点的分类  
  params:  
    labels: 标签列表，原生数组对象   
  return:  
    max_label: 节点确定的分类   
  """

  label_count = {}
  for label in labels:
    label_count[label] = label_count.get(label, 0) + 1
  label_sorted = sorted(label_count.items(), key = operator.itemgetter(1), reverse=True)
  max_label = label_sorted[0][0]
  return max_label

def create_tree(data_set, labels):
  """
  desc:  
    根据数据集和标签名列表生成表示决策树的嵌套字典  
  params:  
    data_set: 数据集，原生数组对象   
    labels: 标签名列表，原生数组对象   
  return:  
    tree: 表示决策树的嵌套字典   
  """

  # 
  # [:]复制数组，以免程序操作修改原数组
  # 
  sub_labels = labels[:]

  label_list = [data[-1] for data in data_set]
  if label_list.count(label_list[0]) == len(label_list):
    return label_list[0]
  if len(data_set[0]) == 1:
    return choose_max_label(label_list)
  max_feature = choose_max_feature(data_set)
  max_feature_name = sub_labels[max_feature]
  tree = {max_feature_name: {}}
  del(sub_labels[max_feature])
  feature_values = [data[max_feature] for data in data_set]
  feature_values = set(feature_values)
  for feature in feature_values:
    tree[max_feature_name][feature] = create_tree(split_data(data_set, max_feature, feature), sub_labels)
  return tree

def classify_tree(feature_vec, tree, features):
  """
  desc:  
    decision tree 预测分类标签  
  params:  
    feature_vec: 待预测特征属性向量，原生数组  
    tree: 表示决策树的嵌套字典   
    features: 特征属性集合，原生数组  
  return:  
    label: 分类标签，字符串   
  """

  feature = list(tree.keys())[0]
  sub_tree = tree[feature]
  feature_index = features.index(feature)
  for key in sub_tree.keys():
    if feature_vec[feature_index] == key:
      if type(sub_tree[key]).__name__ == 'dict':
        label = classify_tree(feature_vec, sub_tree[key], features)
      else:
        label = sub_tree[key]
  return label

def store_tree(tree, file_path):
  """
  desc:  
    存储决策树结构  
  params:  
    tree: 表示决策树的嵌套字典   
    file_path: 决策树存储文件  
  return:  
    None   
  """
  import pickle
  fd = open(file_path, 'wb')
  pickle.dump(tree, fd)
  fd.close()

def grab_tree(file_path):
  """
  desc:  
    读取决策树结构  
  params:  
    file_path: 决策树存储文件  
  return:  
    tree: 表示决策树的嵌套字典   
  """
  import pickle
  fd = open(file_path, 'rb')
  tree = pickle.load(fd)
  return tree

def main():
  print('start decision tree...')
  data_set, labels = file2matrix('data/lenses.txt')
  tree = create_tree(data_set, labels)
  print(tree)
  print(labels)
  tree_plotter.createPlot(tree)

  store_tree(tree, 'data/decision_tree.txt')
  treee = grab_tree('data/decision_tree.txt')
  print(treee)
  tree_plotter.createPlot(tree)

  print('decision tree over...')

if __name__ == '__main__':
  main()
