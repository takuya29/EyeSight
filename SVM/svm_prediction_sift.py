# coding: utf-8
import numpy as np
from sklearn import metrics as met
import csv
from IPython import embed
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

for i in range(15,20):
      f = open("sift_data/train/train_sift_data"+str(i)+".csv","rb")
      list = []
      dataReader=csv.reader(f)
          
      for row in dataReader:
          list.append(row)

      f.close()

      np_train = np.array(list)

      X_train = np_train[:,2:].astype(np.float32)[:]
      y_train = np_train[:,1].astype(np.float32)[:]


      clf = SVC(kernel='linear') #Linear Kernel: Accuracy: Accuracy: 0.772511848341
      #clf = SVC(kernel='rbf',C=1000,gamma = 0.0005) #Accuracy: 0.658767772512 C=1.0, kernel='rbf', degree=3, gamma='auto'
      #clf = SVC(kernel='sigmoid') #Accuracy: 0.658767772512
      #clf = SVC(kernel='poly') #0.78672985782

      clf.fit(X_train,y_train)


      g = open("sift_data/test/test_sift_data"+str(i)+".csv","rb")

      list = []
      dataReader=csv.reader(g)
          
      for row in dataReader:
          list.append(row)

      g.close()

      np_test = np.array(list)

      X_test = np_test[:,2:].astype(np.float32)
      y_test = np_test[:,1].astype(np.float32)


      ans = clf.predict(X_test)

      prob = met.accuracy_score(y_test,ans)

      print i,'Accuracy:',prob

"""
In [14]: ans
Out[14]: 
array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2], dtype=int32)

In [15]: Y_test
Out[15]: 
array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3], dtype=int32)


"""

"""


"""