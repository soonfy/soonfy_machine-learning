#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'preprocess data.'

import pandas as pd
import numpy as np

def clean():
  file = 'data/score.xlsx'
  sheet = 'score'
  print(file)

  # 读取excel
  df = pd.read_excel(file, sheet)

  # 显示数据
  print(df.head())
  print(df.tail(10))

  # 显示数据结构
  print(df.shape)
  print(df.dtypes)


if __name__=="__main__":
  clean()
