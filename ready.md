# ready  

## install python3  
  1. [官网](https://www.python.org/downloads/) 下载特定系统的安装包 或者 brew install python3  
  2. cmd 输入 python3 启动  
  3. python3 搭配 pip3 使用  

## install package  
  1. [numpy](http://www.numpy.org/) pip3 install numpy  
  2. [scipy](https://www.scipy.org/) pip3 install scipy  
  3. [matplotlib](http://matplotlib.org/) pip3 install matplotlib  
  4. [scikit-learn](http://scikit-learn.org/stable/index.html) pip3 install scikit-learn  

## introduction  
  1. the problem setting  
  2. loading an example dataset  
  3. learning and predicting  
  4. model persistence  

## tutorial  
  1. statistical learning  
    * dataset  
    the data needs to be preprocessed in the (n_samples, n_features) shape  
    * estimator  
    fit data, estimator parameters, estimated parameters  
  
  2. supervised learning  
    all supervised estimators: fit method, predict method  
    classification: predict finite labels  
    regression: predict continuous target variable  
    * nearest neighbor  
      KNN k nearest neighbors - 
      given a new observation X_test, find in the training set (i.e. the data used to train the estimator) the observation with the closest feature vector  
    * linear model  
      linear regression - 
      fits a linear model to the data set by adjusting a set of parameters in order to make the sum of the squared residuals of the model as small as possible  
      Y = a * X + b, Y: target variable, X: data, a: coefficients, b: observation noise  
    * support vector machines  
      linear SVMs - 
      find a combination of samples to build a plane maximizing the margin between the two classes  

  3. model selection  
    * score  
    * cross-validated estimators  

  4. unsupervised learning  
    * clustering  
      k-means clustering  
    * decompositions  
      principal component analysis  
      independent component analysis  
      