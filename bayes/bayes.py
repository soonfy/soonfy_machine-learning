#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@module bayes
@methods
  1. gene_vocab_list
  2. word2vec
  3. train_bayes
  4. classify_bayes

@desc study bayes
@author soonfy<soonfy@163.com>
@log
created at: 2017-11-16
"""
print(__doc__)

import numpy as np

def loadDataSet():
  """
  创建数据集
  :return: 单词列表postingList, 所属类别classVec
  """
  postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                  ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
  classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
  return postingList, classVec

def gene_vocab_list(data_set):
  """
  desc:  
    获取所有数据集中单词的集合
  params:  
    data_set: 文本数据集，原生数组   
  return:  
    vocab_list: 单词集合，原生数组    
  """

  vocab_set = set([])
  for doc in data_set:
    # 
    # 技巧
    # | 求集合的并集
    # 
    vocab_set = vocab_set | set(doc)
  vocab_list = list(vocab_set)
  return vocab_list
  
def word2vec(data_set, vocab_list):
  """
  desc:  
    获取数据集的词向量
  params:  
    data_set: 文本数据集，原生数组   
    vocab_list: 单词集合，原生数组    
  return:  
    word_vec: 单词向量，原生数组    
  """

  word_vec = [0] * len(vocab_list)
  for word in data_set:
    if word in vocab_list:
      # 
      # 词集模型
      # word_vec[vocab_list.index(word)] = 1
      # 词袋模型
      word_vec[vocab_list.index(word)] += 1
    else:
      print('word %s isnot in vocab set.' % word)
  return word_vec

def train_bayes(vocab_mat, label_vec):
  """
  desc:  
    训练数据集
  params:  
    vocab_mat: 词向量集合，原生数组   
    label_vec: 类别集合，原生数组    
  return:  
    pro_pos_vec: 在0类别下，每个单词出现的概率    
    pro_neg_vec: 在1类别下，每个单词出现的概率    
    pro_neg: 类别1出现的概率    
  """

  len_data = len(vocab_mat)
  len_vocab = len(vocab_mat[0])
  pro_neg = sum(label_vec) / float(len_data)
  # 
  # 避免列表中某个单词为0，最后乘积为0，默认改为1
  # 总数统计默认改为2
  # 
  # pros_pos, pros_neg = np.zeros(len_vocab), np.zeros(len_vocab)
  # sum_pos, sum_neg = 0.0, 0.0
  pros_pos, pros_neg = np.ones(len_vocab), np.ones(len_vocab)
  sum_pos, sum_neg = 2.0, 2.0
  for index in range(len_data):
    if label_vec[index] == 1:
      pros_neg += vocab_mat[index]
      sum_neg += sum(vocab_mat[index])
    else:
      pros_pos += vocab_mat[index]
      sum_pos += sum(vocab_mat[index])
  # 
  # 避免数组太小溢出，取对数
  # 
  # pro_neg_vec = pros_neg / sum_neg
  # pro_pos_vec = pros_pos / sum_pos
  pro_neg_vec = np.log(pros_neg / sum_neg)
  pro_pos_vec = np.log(pros_pos / sum_pos)
  return pro_pos_vec, pro_neg_vec, pro_neg

def classify_bayes(feature_vec, pro_pos_vec, pro_neg_vec, pro_neg):
  """
  desc:  
    测试数据集
  params:  
    feature_vec: 词向量集合，原生数组   
    pro_pos_vec: 在0类别下，每个单词出现的概率    
    pro_neg_vec: 在1类别下，每个单词出现的概率    
    pro_neg: 类别1出现的概率    
  return:  
    label: 预测出数据集的所属类别    
  """

  pro_pos_v = sum(feature_vec * pro_pos_vec) + np.log(1 - pro_neg)
  pro_neg_v = sum(feature_vec * pro_neg_vec) + np.log(pro_neg)
  print(pro_pos_v, pro_neg_v)
  if pro_pos_v > pro_neg_v:
    label = 0
  else:
    label = 1
  return label




postingList, classVec = loadDataSet()
vocab_list = gene_vocab_list(postingList)
print(vocab_list)
result = []
for data in postingList:
  result.append(word2vec(data, vocab_list))

pro_pos_vec, pro_neg_vec, pro_neg = train_bayes(result, classVec)
print(pro_pos_vec)
print(pro_neg_vec)
print(pro_neg)

words = ['love', 'my', 'dalmation']
# words = ['stupid', 'garbage']
feature_vec = np.array(word2vec(words, vocab_list))
print(classify_bayes(feature_vec, pro_pos_vec, pro_neg_vec, pro_neg))
