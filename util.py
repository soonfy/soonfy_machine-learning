#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

# log func
def log(func):
  def wrapper(*args, **kw):
    print('call %s():' % func.__name__)
    return func(*args, **kw)
  return wrapper

# file_path
# read line length
@log
def read_file(file_path, length=100):
  print('file path -->', file_path)
  data = []
  try:
    with open(file_path, 'r') as f:
      if length > 0:
        str_data = ''
        for x in range(1, length + 2):
          line = f.readline()
          str_data += line
        data.append(str_data.strip())
      else:
        data = f.readlines().strip()
  except Exception as error:
    print('error')
    print(error)
  finally:
    # print(data)
    return data


# dir_path
# file ext
@log
def read_dir(dir_path, ext='.txt'):
  print('dir path -->', dir_path)
  # print([
  #   x for x in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, x)) and os.path.splitext(x)[1] == ext
  # ])

# name, followed, following, sexal
@log
def pre_data(data_arr):
  data = []
  for lines in data_arr:
    for line in lines.split('\n'):
      temp = line.split('\t')
      data.append([temp[8], temp[5], temp[6], temp[2], temp[3], temp[4]])
  # print(data)
  return data

@log
def draw_scatter(data_arr):
  data_arr = data_arr[1:]
  datax = []
  datay = []
  datat = []
  for temp in data_arr:
    if int(temp[1]) < 300 and int(temp[2]) < 250:
      datax.append(temp[1])
      datay.append(temp[2])
      datat.append(0)
    elif int(temp[1]) < 300 and (int(temp[2]) > 500 and int(temp[2]) < 1000):
      datax.append(temp[1])
      datay.append(temp[2])
      datat.append(1)
    elif (int(temp[1]) > 500 and int(temp[1]) < 1000) and (int(temp[2]) > 500 and int(temp[2]) < 1000):
      datax.append(temp[1])
      datay.append(temp[2])
      datat.append(3)
    elif (int(temp[1]) > 500 and int(temp[1]) < 1000) and int(temp[2]) < 250:
      datax.append(temp[1])
      datay.append(temp[2])
      datat.append(4)
  
  print(len(datax))
  
  plt.title('scatter')
  # print(datax)
  # print(datay)
  plt.xlabel(u'followed')
  plt.ylabel('following')
  plt.xlim(0, 1200)
  plt.ylim(0, 1200)

  datax = np.array(datax, 'int32')
  datay = np.array(datay, 'int32')
  datat = np.array(datat, 'int32')

  sizes = []
  colors = []
  linewidths = []
  edgecolors = []
  for temp in datat:
    if temp == 1:
      colors.append('#4AB8EA')
      sizes.append(20)
      linewidths.append(0.5)
      # edgecolors.append('#E94F5A')
    elif temp == 2:
      colors.append('#E2769F')
    elif temp == 3:
      colors.append('#E94F5A')
    elif temp == 4:
      colors.append('#F5C224')
    else:
      colors.append('#4AB8EA')
      sizes.append(30)
      linewidths.append(1)
      # edgecolors.append('#F18754')
  
  marker = 'o'
  alpha = 0.5

  plt.scatter(datax, datay, s = sizes, c = colors, marker = marker, alpha = alpha, linewidths = linewidths, edgecolors = edgecolors)
  
  plt.show()

@log
def draw_fill(data_arr):
  data_arr = data_arr[1:]
  datax = []
  datay = []
  for temp in data_arr:
    if int(temp[1]) < 2000 and int(temp[2]) < 2000:
      datax.append(temp[1])
      datay.append(temp[2])
  
  plt.title('fill between')
  plt.xlabel(u'followed')
  plt.ylabel('following')
  plt.xlim(0, 200)
  plt.ylim(0, 200)

  datay1 = np.array(datax, 'int32')[0:200]
  datay2 = np.array(datay, 'int32')[0:200]
  datax = np.arange(0, 200)

  plt.fill_between(datax, datay1, datay2, where=(datay1>=datay2), color='red', alpha=0.25)
  plt.fill_between(datax, datay1, datay2, where=(datay1<datay2), color='green', alpha=0.25)

  plt.show()

@log
def draw_bar(data_arr):
  data_arr = data_arr[1:]
  datax = []
  datay = []
  for temp in data_arr:
    if int(temp[1]) < 2000 and int(temp[2]) < 2000:
      datax.append(temp[1])
      datay.append(temp[2])
  
  plt.title('bar')
  plt.xlabel(u'followed')
  plt.ylabel('following')
  plt.xlim(0, 2000)
  plt.ylim(0, 2000)

  datax = np.array(datax, 'int32')
  datay = np.array(datay, 'int32')

  plt.show()


data = read_file('./data/weibo-2017-10-01.txt', 1)
print(data)
data = pre_data(read_file('./data/weibo-2017-10-01.txt', 10000))
draw_scatter(data)
# draw_fill(data)
