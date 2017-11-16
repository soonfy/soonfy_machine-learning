# decision tree

### k-近邻算法    
* 优点：精度高，对异常值不敏感，无数据输入假定。    
* 缺点：计算复杂度高，空间复杂度高。    
* 适用数据范围：数值型，标称型。    

### 一般流程    
1. 收集数据：可以使用任何方法。    
2. 准备数据：距离计算所需要的数值，最好是结构化的数据格式。    
3. 分析数据：可以使用任何方法。    
4. 训练算法：此步骤不适用于k-近邻算法。    
5. 测试算法：计算错误率。    
6. 适用算法：首先需要输入样本数据和结构化的输出结果，然后运行k-近邻算法判定输入数据分别属于哪个分类，最后应用对计算出的分类执行后续的处理。   

### dating数据集    
1. 第 1 列：年龄    
2. 第 2 列：玩视频游戏所耗时间百分比    
3. 第 3 列：每周消费的冰淇淋公升数    
4. 第 4 列：类别。1 表示不喜欢， 2 表示魅力一般， 3 表示极具魅力    

### scikit-learn decision tree

  ```
  from sklearn import neighbors
  neigh = neighbors.KNeighborsClassifier(n_neighbors, weights)
  neigh.fit(attr_mat, label_vec)
  labels = neigh.predict(test_mat)
  ```

  1. KNeighborsClassifier 参数    
  sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)    
  * n_neighbors: 最近邻个数。    
  * weights: 距离权重计算方式。 uniform 表示距离权重相同，distance 表示权重和距离成反比。    
  * algorithm: 数据集分类方法。 ball_tree ，kd_tree 和暴力搜索。    
  * leaf_size: 叶子节点大小，适用于 ball_tree 和 kd_tree。     
  * p: 距离计算方式。 1 曼哈顿距离，2 欧几里得距离，n 闵可夫斯基距离。    
  * metric: 距离度量方式。 minkowski 表示闵可夫斯基距离，
  * metric_params: 距离度量参数。    
  * n_jobs: 任务并发数量。 -1 表示 CPU 核数。    

  2. neigh 对象的方法    
  * fit(attr_mat, label_vec): 训练模型    
  * get_params(): 返回分类器参数    
  * kneighbors(attr_mat, n_neighbors, return_distance): 返回邻近数据对象    
  * kneighbors_graph(attr_mat): 返回数据集连接矩阵    
  * predict(attr_mat): 预测数据对象    
  * predict_proba(attr_mat): 预测数据对象的概率    
  * score(attr_mat, label_vec): 测试分类效果    
  * set_params(): 配置分类器参数    
