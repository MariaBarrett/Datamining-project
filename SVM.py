from __future__ import division
from collections import Counter
from featuremap import featuremap, metadatamap
import datasplit
import pickle
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
import numpy as np

y_train, X_train = datasplit.natlan_train[0], datasplit.natlan_train[1]
y_test, X_test = datasplit.natlan_test[0], datasplit.natlan_test[1]

parameters = {'C':[1, 3, 5, 7, 9, 11], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(X_train, y_train)
print clf.best_params_