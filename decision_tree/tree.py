#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@module tree
@methods
  1. calculate_shannon_entropy
  2. split_data
  3. choose_feature_split
  4. choose_max_label
  5. calculate_shannon_entropy

@desc study decision tree
@author soonfy<soonfy@163.com>
@log
created at: 2017-11-10
"""
print(__doc__)

import math
import operator


def createDataSet():
    """DateSet 基础数据集
    Args:
        无需传入参数
    Returns:
        返回数据集和对应的label标签
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

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

  label_list = [data[-1] for data in data_set]
  if label_list.count(label_list[0]) == len(label_list):
    return label_list[0]
  if len(data_set[0]) == 1:
    return choose_max_label(label_list)
  max_feature = choose_max_feature(data_set)
  max_feature_name = labels[max_feature]
  tree = {max_feature_name: {}}
  del(labels[max_feature])
  feature_values = [data[max_feature] for data in data_set]
  feature_values = set(feature_values)
  for feature in feature_values:
    sub_labels = labels[:]
    tree[max_feature_name][feature] = create_tree(split_data(data_set, max_feature, feature), sub_labels)
  return tree



data_set, labels = createDataSet()
print(calculate_shannon_entropy(data_set))
max_feature = choose_max_feature(data_set)
print(max_feature)
tree = create_tree(data_set, labels)
print(tree)
