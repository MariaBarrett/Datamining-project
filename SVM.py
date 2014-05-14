from __future__ import division
from collections import Counter
from featuremap import featuremap, metadatamap
import datasplit
import clean
import pickle
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

metadata = pickle.load( open( "metadata.p", "rb" ) )
data = pickle.load( open( "dataset.p", "rb" ) )
plusone = np.where(metadata[:,9] != "1")[0] # Get indexes for all texts written by more than one person
data, metadata = np.delete(data, plusone, 0), np.delete(metadata, plusone, 0) # Remove all texts written by more than one person from data

# RAW DATA SETS
NLtrain_y, NLtrain_X, NLtest_y, NLtest_X = datasplit.natlan(metadata,data)
GRtrain_y, GRtrain_X, GRtest_y, GRtest_X = datasplit.grade(metadata,data)
LEtrain_y, LEtrain_X, LEtest_y, LEtest_X = datasplit.level(metadata,data)
AUtrain_y, AUtrain_X, AUtest_y, AUtest_X = datasplit.author(metadata,data,4,3,2)

# NORMALIZED DATA SETS
NLtrain_Xn, NLtest_Xn = clean.normalize(NLtrain_X, NLtest_X)
GRtrain_Xn, GRtest_Xn = clean.normalize(GRtrain_X, GRtest_X)
LEtrain_Xn, LEtest_Xn = clean.normalize(LEtrain_X, LEtest_X)
AUtrain_Xn, AUtest_Xn = clean.normalize(AUtrain_X, AUtest_X)

"""
#Calling PCA functions
clean.princomp(NLtrain_Xn)
NLtrain_Xn_pca, NLtest_Xn_pca = clean.princomp_transform(NLtrain_Xn, NLtest_Xn, 100)
clean.princomp(GRtrain_Xn)
GRtrain_Xn_pca, GRtest_Xn_pca = clean.princomp_transform(GRtrain_Xn, GRtest_Xn, 100)
clean.princomp(GRtrain_Xn)
LEtrain_Xn_pca, LEtest_Xn_pca = clean.princomp_transform(LEtrain_Xn, LEtest_Xn, 100)
clean.princomp(GRtrain_Xn)
AUtrain_Xn_pca, AUtest_Xn_pca = clean.princomp_transform(AUtrain_Xn, AUtest_Xn, 100)
"""
#----------------------------------------------------------------------------------------
#SVM
#----------------------------------------------------------------------------------------
def SVM(X_train, y_train, X_test, y_test, subset=False, best_features = []):
	parameters = {'C': [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1,]}
	svr = svm.SVC()
	clf = grid_search.GridSearchCV(svr, parameters)

	print '-'*45
	print "All features"
	print '-'*45	

	clf.fit(X_train, y_train)
	print clf.best_params_
	print "0-1 loss", clf.score(X_test, y_test)

	print "Best parameters: ", clf.best_params_

	print "Detailed classification report:"
	print ""
	print "The model is trained on the full development set."
	print "The scores are computed on the full evaluation set."
	print ""
	y_true, y_pred = y_test, clf.predict(X_test)
	print classification_report(y_true, y_pred)
	print ""


	if subset:
		subsets = ['LEX', 'WB', 'SYN', 'STRUC']
		for subset in subsets: 
			print '-'*45
			print subset
			print '-'*45
			clf.fit(X_train[:,datasplit.sub(subset)], y_train) #tjek gerne at det er rigtigt. Shapen ser rigtig ud. Jeg er bare lidt traet pt. 
			print clf.best_params_
			print "0-1 loss", clf.score(X_test[:,datasplit.sub(subset)], y_test)

			print "Best parameters: ", clf.best_params_

			print "Detailed classification report:"
			print ""
			print "The model is trained on the full development set."
			print "The scores are computed on the full evaluation set."
			print ""
			y_true, y_pred = y_test, clf.predict(X_test[:,datasplit.sub(subset)])
			print classification_report(y_true, y_pred)
			print ""

	if len(best_features) > 1:
		print '-'*45
		print "With %s best features" %(len(best_features))
		print '-'*45	

		clf.fit(X_train[:,NLsorted_indices_of_best_features], y_train)
		print clf.best_params_
		print "0-1 loss", clf.score(X_test[:,NLsorted_indices_of_best_features], y_test)

		print "Best parameters: ", clf.best_params_

		print "Detailed classification report:"
		print ""
		print "The model is trained on the full development set."
		print "The scores are computed on the full evaluation set."
		print ""
		y_true, y_pred = y_test, clf.predict(X_test[:,NLsorted_indices_of_best_features])
		print classification_report(y_true, y_pred)
		print ""

#---------------------------------------------------------------
#Calling

print "*"*45
print "Native Language"
print "*"*45

#Calling feature selection
datasplit.inspect_tree_selection(NLtrain_Xn, NLtrain_y, "Native language")
NLsorted_indices_of_best_features = datasplit.tree_selection(NLtrain_Xn, NLtrain_y, 20)

SVM(NLtrain_Xn, NLtrain_y, NLtest_Xn, NLtest_y, True, NLsorted_indices_of_best_features)

"""
print "*"*45
print "Grade"
print "*"*45
SVM(GRtrain_Xn, GRtrain_y, GRtest_Xn, GRtest_y)

datasplit.inspect_tree_selection(GRtrain_Xn, GRtrain_y, "Grade")
LEsorted_indices_of_best_features = datasplit.tree_selection(GRtrain_Xn, GRtrain_y, 20)
"""

print "*"*45
print "Level"
print "*"*45

datasplit.inspect_tree_selection(LEtrain_Xn, LEtrain_y, "Academic level")
LEsorted_indices_of_best_features = datasplit.tree_selection(LEtrain_Xn, LEtrain_y, 20)

SVM(LEtrain_Xn, LEtrain_y, LEtest_Xn, LEtest_y, True, LEsorted_indices_of_best_features)
