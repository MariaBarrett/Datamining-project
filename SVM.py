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

# TREE SELECTION INDICES
#datasplit.inspect_tree_selection(NLtrain_Xn, NLtrain_y, "Native language")
#NLtree_selection = datasplit.tree_selection(NLtrain_Xn, NLtrain_y, 20)
#datasplit.inspect_tree_selection(GRtrain_Xn, GRtrain_y, "Grade")
#GRtree_selection = datasplit.tree_selection(GRtrain_Xn, GRtrain_y, 20)
#datasplit.inspect_tree_selection(LEtrain_Xn, LEtrain_y, "Academic level")
#LEtree_selection = datasplit.tree_selection(LEtrain_Xn, LEtrain_y, 20)
#datasplit.inspect_tree_selection(AUtrain_Xn, AUtrain_y, "Author")
#AUtree_selection = datasplit.tree_selection(AUtrain_Xn, AUtrain_y, 20)

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
# SVM MAIN FUNCTION
#----------------------------------------------------------------------------------------

def SVM(X_train, y_train, X_test, y_test, subsets, tree_select=False, anova=False):
	
	parameters = {'C': [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1,]}
	svr = svm.SVC()
	clf = grid_search.GridSearchCV(svr, parameters)
	
	if tree_select:
		subsets.append(["TREE SELECTION"])
	if anova:
		subsets.append(["ANOVA"])

	for ss in subsets:

		print '-'*45
		print '+'.join(ss)+" features"
		print '-'*45+"\n"	

		if ss == ["TREE SELECTION"]:
			feat_index = datasplit.tree_selection(X_train, y_train, tree_select)
		elif ss == ["ANOVA"]:
			feat_index = datasplit.anova(X_train, y_train, anova) # Virker ikke... ENDNU!
		else: 
			feat_index = datasplit.sub(ss)

		a = X_test[:,feat_index]
		
		print a[375]

		
		clf.fit(X_train[:,feat_index], y_train)
		print "0-1 loss", clf.score(X_test[:,feat_index], y_test)
		print "Best parameters: ", clf.best_params_

		print "\nDetailed classification report:\n"
		print "The model is trained on the full development set."
		print "The scores are computed on the full evaluation set."
		print ""
		y_true, y_pred = y_test, clf.predict(X_test[:,feat_index])
		print classification_report(y_true, y_pred)
		print ""

#----------------------------------------------------------------------------------------
# EXPERIMENTS
#----------------------------------------------------------------------------------------
"""
print "\n"+"*"*45
print "Native Language"
print "*"*45+"\n"
SVM(NLtrain_Xn, NLtrain_y, NLtest_Xn, NLtest_y,
	[["F1"],["F2"],["F3"],["F4"],["F1","F2"],["F1","F2","F3"],["all"]], tree_select=200)
"""
print "\n"+"*"*45
print "Grade"
print "*"*45+"\n"
SVM(GRtrain_Xn, GRtrain_y, GRtest_Xn, GRtest_y,
	[["F2"]])
"""
print "\n"+"*"*45
print "Academic level"
print "*"*45+"\n"
SVM(LEtrain_Xn, LEtrain_y, LEtest_Xn, LEtest_y,
	[["F1"],["F2"],["F3"],["F4"],["F1","F2"],["F1","F2","F3"],["all"]], tree_select=200)

print "\n"+"*"*45
print "Author"
print "*"*45+"\n"
SVM(AUtrain_Xn, AUtrain_y, AUtest_Xn, AUtest_y,
	[["F1"],["F2"],["F3"],["F4"],["F1","F2"],["F1","F2","F3"],["all"]], tree_select=150)
"""
