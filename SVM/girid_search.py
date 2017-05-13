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

f = open("train_data.csv","rb")
list = []
dataReader=csv.reader(f)
    
for row in dataReader:
    list.append(row)

f.close()

np_train = np.array(list)

X = np_train[:,2:].astype(np.int32)
y = np_train[:,1].astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000,100000000000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

#clf = SVC(kernel='linear',probability=True) #Linear Kernel: Accuracy: Accuracy: 0.772511848341
#clf = SVC(kernel='rbf',probability=True) #Accuracy: 0.658767772512 C=1.0, kernel='rbf', degree=3, gamma='auto'
#clf = SVC(kernel='sigmoid') #Accuracy: 0.658767772512
#clf = SVC(kernel='poly') #0.78672985782

score = 'precision'
clf = GridSearchCV(
    SVC(C=1), # 識別器
    tuned_parameters, # 最適化したいパラメータセット 
    cv=5, # 交差検定の回数
    n_jobs = -1,
    scoring='%s_weighted' % score ) # モデルの評価関数の指定

clf.fit(X_train,y_train)

"""
g = open("test_data.csv","rb")

list = []
dataReader=csv.reader(g)
    
for row in dataReader:
    list.append(row)

g.close()

np_test = np.array(list)

X_test = np_test[:,2:].astype(np.int32)
Y_test = np_test[:,1].astype(np.int32)


ans = clf.predict(X_test)

prob = met.accuracy_score(Y_test,ans)

print 'Accuracy:',prob
"""
embed()

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