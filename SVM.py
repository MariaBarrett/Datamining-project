from __future__ import division
from collections import Counter
from featuremap import featuremap, metadatamap
import pickle
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
import numpy as np

metadata = pickle.load( open( "metadata.p", "rb" ) )
labels = np.copy(metadata[:,-4])
traindata = pickle.load( open( "dataset.p", "rb" ) )
print labels
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
#clf.fit(traindata, labels)