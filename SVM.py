from __future__ import division
from collections import Counter
from featuremap import featuremap, metadatamap
import datasplit
import clean
import pickle
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

y_train, X_train = datasplit.GRtrain_y, datasplit.GRtrain_X
y_test, X_test = datasplit.GRtest_y, datasplit.GRtest_X

X_train, X_test = clean.normalize(X_train, X_test)

#----------------------------------------------------------------------------------------
#PCA
#----------------------------------------------------------------------------------------

def princomp(train):
	print "#" * 45
	print "PCA"
	print "#" * 45
	pca = PCA(copy=True)
	transformed = pca.fit_transform(train)
	components = pca.components_
	exp_variance = pca.explained_variance_ratio_

	x = np.array([i for i in range(len(exp_variance))])

	#print "Explained variance", exp_variance
	M = (train-np.mean(train.T,axis=1)).T # subtract the mean (along columns)
 	eigv, eigw = np.linalg.eig(np.cov(M)) 
	plt.plot(x, exp_variance)
	plt.title("Explained Variance")
	plt.ylabel("Exp. variance")
	plt.xlabel("Components")
	plt.show()

def princomp_transform(trainset, testset, components):
	#print trainset[0]
	pca = PCA(n_components=components, copy=True, whiten=False)
	pca.fit(trainset)
	X_train_trans = pca.transform(trainset)
	X_test_trans = pca.transform(testset)
	#orig = pca.inverse_transform(X_train_trans)
	#print orig[0]
	print "PCA transformation done"
	return X_train_trans, X_test_trans

princomp(X_train)
X_train_PCA, X_test_PCA = princomp_transform(X_train, X_test, 100)

#----------------------------------------------------------------------------------------
#KNN
#----------------------------------------------------------------------------------------
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)


#----------------------------------------------------------------------------------------
#SVM
#----------------------------------------------------------------------------------------
print "#" * 45
print "SVM"
print "#" * 45
	
parameters = {'C':[0.0001, 0.001, 0.1 ,1, 3, 5, 7, 11, 13 ], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1,]}
"""
svr = svm.SVC()

clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(X_train_PCA, y_train)
print clf.best_params_
print clf.score(X_test_PCA, y_test)


"""

scores = ['precision', 'recall']

for score in scores:

    clf = GridSearchCV(svm.SVC(), parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print "Best parameters set found on development set:"
    print ""
    print clf.best_estimator_
    print ""
    print "Grid scores on development set:" 
    print ""
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print ""
    print "Best parameters: ", clf.best_params_

    print "Detailed classification report:"
    print ""
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print ""
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, y_pred)
    print ""
