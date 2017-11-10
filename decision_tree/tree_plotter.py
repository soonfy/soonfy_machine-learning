#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@module plot tree
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
  root = tree.keys()[0]
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
  root = tree.keys()[0]
  super_leaf = tree[root]
  for sub_leaf in super_leaf.keys():
    if type(super_leaf[sub_leaf]).__name__ == 'dict':
      this_depth = 1 + get_depth(super_leaf[sub_leaf])
    else:
      this_depth = 1
    if this_depth > depth:
      depth = this_depth
  return depth

