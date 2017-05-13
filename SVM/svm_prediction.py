# coding: utf-8
import numpy as np
from sklearn import metrics
import csv
from IPython import embed
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.svm import SVC

dname = "not_hist"

f = open(dname+"/train_data.csv","rb")
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


g = open(dname+"/test_data.csv","rb")

list = []
dataReader=csv.reader(g)
    
for row in dataReader:
    list.append(row)

g.close()

np_test = np.array(list)

X_test = np_test[:,2:].astype(np.float32)
y_test = np_test[:,1].astype(np.float32)


ans = clf.predict(X_test)

prob = metrics.accuracy_score(y_test,ans)
metrics.confusion_matrix(y_test,ans)

print 'Accuracy:',prob
