import numpy as np

import utiler
from knn import knn
from sklearn.neighbors import KNeighborsClassifier

def main():
  data = utiler.read_file('./data/weibo-2017-10-01.txt', 1)
  print(data)
  data = utiler.pre_data(utiler.read_file('./data/weibo-2017-10-01.txt', 10000))
  data = data[1:]
  
  datax = []
  datay = []

  data_sets = []
  labels = []
  data_tes = []
  for temp in data:
    if int(temp[1]) < 300 and int(temp[2]) < 250:
      datax.append(temp[1])
      datay.append(temp[2])

      data_sets.append([temp[1], temp[2]])
      labels.append(1)
    elif int(temp[1]) < 300 and (int(temp[2]) > 500 and int(temp[2]) < 1000):
      datax.append(temp[1])
      datay.append(temp[2])

      data_sets.append([temp[1], temp[2]])
      labels.append(2)
    elif (int(temp[1]) > 500 and int(temp[1]) < 1000) and (int(temp[2]) > 500 and int(temp[2]) < 1000):
      datax.append(temp[1])
      datay.append(temp[2])

      data_sets.append([temp[1], temp[2]])
      labels.append(3)
    elif (int(temp[1]) > 500 and int(temp[1]) < 1000) and int(temp[2]) < 250:
      datax.append(temp[1])
      datay.append(temp[2])

      data_sets.append([temp[1], temp[2]])
      labels.append(4)
    elif (int(temp[1]) > 300 and int(temp[1]) < 500) and (int(temp[2]) > 250 and int(temp[2]) < 1000):
      data_tes.append([temp[1], temp[2]])

  datax = np.array(datax, 'int32')
  datay = np.array(datay, 'int32')
  data_sets = np.array(data_sets, 'int32')
  labels = np.array(labels, 'int32')
  data_tes = np.array(data_tes, 'int32')

  print(len(datax))
  print(len(data_sets))
  print(len(labels))
  print(len(data_tes))

  datax_new = []
  datay_new = []
  label_new = []

  neigh = KNeighborsClassifier(n_neighbors=10)
  neigh.fit(data_sets, labels)

  for index in range(10):
    label = knn.classify(data_tes[index], data_sets, labels, 10)
    label2 = neigh.predict([data_tes[index]])
    print(data_tes[index], 'study', label, 'sklearn', label2)
    datax_new.append(data_tes[index][0])
    datay_new.append(data_tes[index][1])
    label_new.append(label)

  datax = np.append(datax, datax_new)
  datay = np.append(datay, datay_new)
  labels = np.append(labels, label_new)

  utiler.draw_scatter(datax, datay, labels)

if __name__ == '__main__':
  main()
