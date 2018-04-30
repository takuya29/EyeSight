from __future__ import print_function

import csv

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

f = open("./input/sift_data/train/train_sift_data5.csv", "rb")
list = []
dataReader = csv.reader(f)

for row in dataReader:
    list.append(row)

f.close()

np_train = np.array(list)

X = np_train[:, 2:].astype(np.float32)[:]
y = np_train[:, 1].astype(np.float32)[:]

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4],
     'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
]

score = 'f1'

clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                   scoring='%s_macro' % score)
clf.fit(X_train, y_train)

from sklearn.externals import joblib

joblib.dump('output/svm_model.pkl', clf.best_estimator_)

print(clf.best_params_, clf.best_score_)
