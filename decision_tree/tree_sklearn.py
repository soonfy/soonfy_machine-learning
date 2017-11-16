#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@module tree sklearn
@methods
  1. file2matrix


@desc study scikit-learn decision tree
@author soonfy<soonfy@163.com>
@log
created at: 2017-11-15
"""
print(__doc__)

from sklearn.tree import DecisionTreeClassifier
import tree as tree

def classify_tree_sk(feature_mat, label_vec, feature_vec):
  dtree = DecisionTreeClassifier()
  dtree.fit(feature_mat, label_vec)
  label = dtree.predict(feature_vec)
  return label

def main():
  data_set, label_names = tree.file2matrix('data/lenses.txt')
  len_data = len(data_set)
  feature_mat = []; labels = []
  for index in range(len_data):
    feature_mat.append(data_set[index][0:4])
    labels.append(data_set[index][-1:])
  print(classify_tree_sk(feature_mat, labels, ['young',	'myope',	'no',	'reduced']))

if __name__ == '__main__':
  main()

  
