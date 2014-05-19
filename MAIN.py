from __future__ import division
from collections import Counter
from featuremap import featuremap, metadatamap
import datasplit
import clean
import pickle
import matplotlib.pyplot as plt
from sklearn import svm, tree, neighbors, grid_search
import numpy as np
from scipy import stats
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
AUtrain_y20, AUtrain_X20, AUtest_y20, AUtest_X20 = datasplit.author(metadata,data,4,20,2)
AUtrain_y100, AUtrain_X100, AUtest_y100, AUtest_X100 = datasplit.author(metadata,data,4,100,2)
AUtrain_yAH, AUtrain_XAH, AUtest_yAH, AUtest_XAH = datasplit.author(metadata,data,4,20,2,(7,"AH"))

# NORMALIZED DATA SETS
NLtrain_Xn, NLtest_Xn = clean.normalize(NLtrain_X, NLtest_X)
GRtrain_Xn, GRtest_Xn = clean.normalize(GRtrain_X, GRtest_X)
LEtrain_Xn, LEtest_Xn = clean.normalize(LEtrain_X, LEtest_X)
AUtrain_Xn20, AUtest_Xn20 = clean.normalize(AUtrain_X20, AUtest_X20)
AUtrain_Xn100, AUtest_Xn100 = clean.normalize(AUtrain_X100, AUtest_X100)
AUtrain_XnAH, AUtest_XnAH = clean.normalize(AUtrain_XAH, AUtest_XAH)


# TREE SELECTION FEATURES EVALUATION PLOTS
#datasplit.inspect_tree_selection(NLtrain_Xn, NLtrain_y, "Native language")
#datasplit.inspect_tree_selection(GRtrain_Xn, GRtrain_y, "Grade")
#datasplit.inspect_tree_selection(LEtrain_Xn, LEtrain_y, "Academic level")
#datasplit.inspect_tree_selection(AUtrain_Xn20, AUtrain_y20, "Author")
#datasplit.inspect_tree_selection(AUtrain_Xn100, AUtrain_y100, "Author")
#datasplit.inspect_tree_selection(AUtrain_XnAH, AUtrain_yAH, "Author")

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

def main(classifier, X_train, y_train, X_test, y_test, subsets, tree_select=False, anova=False):
	
	if classifier == "SVM":
		parameters = {'C': [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1,]}
		svr = svm.SVC()
		clf = grid_search.GridSearchCV(svr, parameters)
	elif classifier == "DTC":
		parameters = {'max_depth': [1, 2, 4, 8, 16, 32, 64, None], 'min_samples_leaf':[1, 2, 4, 8, 16, 32]}
		id3 = tree.DecisionTreeClassifier()
		clf = grid_search.GridSearchCV(id3, parameters)
	elif classifier == "KNN":
		parameters = {'n_neighbors': [1, 3, 5, 9, 13, 17, 35, 65]}
		knn = neighbors.KNeighborsClassifier()
		clf = grid_search.GridSearchCV(knn, parameters)
	else:
		print "Unable to recognize the requested algorithm"
	
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
		
		clf.fit(X_train[:,feat_index], y_train)
		print "Best parameters: ", clf.best_params_
		print "\n0-1 loss", clf.score(X_test[:,feat_index], y_test)
		print "Baseline", stats.mode(y_test)[1][0]/len(y_test) # ZeroR baseline

		print "\nDetailed classification report:\n"
		print ""
		y_true, y_pred = y_test, clf.predict(X_test[:,feat_index])
		print classification_report(y_true, y_pred)
		print ""

#----------------------------------------------------------------------------------------
# EXPERIMENTS
#----------------------------------------------------------------------------------------

print "\n"+"*"*45
print "Native Language"
print "*"*45+"\n"
main("SVM", NLtrain_Xn, NLtrain_y, NLtest_Xn, NLtest_y,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=225)
main("DTC", NLtrain_Xn, NLtrain_y, NLtest_Xn, NLtest_y,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=225)
main("KNN", NLtrain_Xn, NLtrain_y, NLtest_Xn, NLtest_y,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=225)

print "\n"+"*"*45
print "Grade"
print "*"*45+"\n"
main("SVM", GRtrain_Xn, GRtrain_y, GRtest_Xn, GRtest_y,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=215)
main("DTC", GRtrain_Xn, GRtrain_y, GRtest_Xn, GRtest_y,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=215)
main("KNN", GRtrain_Xn, GRtrain_y, GRtest_Xn, GRtest_y,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=215)

print "\n"+"*"*45
print "Academic level"
print "*"*45+"\n"
main("SVM", LEtrain_Xn, LEtrain_y, LEtest_Xn, LEtest_y,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=200)
main("DTC", LEtrain_Xn, LEtrain_y, LEtest_Xn, LEtest_y,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=200)
main("KNN", LEtrain_Xn, LEtrain_y, LEtest_Xn, LEtest_y,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=200)

print "\n"+"*"*45
print "Author: 20"
print "*"*45+"\n"
main("SVM", AUtrain_Xn20, AUtrain_y20, AUtest_Xn20, AUtest_y20,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=180)
main("DTC", AUtrain_Xn20, AUtrain_y20, AUtest_Xn20, AUtest_y20,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=180)
main("KNN", AUtrain_Xn20, AUtrain_y20, AUtest_Xn20, AUtest_y20,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=180)

print "\n"+"*"*45
print "Author: 20, Art & Humanities"
print "*"*45+"\n"
main("SVM", AUtrain_XnAH, AUtrain_yAH, AUtest_XnAH, AUtest_yAH,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=200)
main("DTC", AUtrain_XnAH, AUtrain_yAH, AUtest_XnAH, AUtest_yAH,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=200)
main("KNN", AUtrain_XnAH, AUtrain_yAH, AUtest_XnAH, AUtest_yAH,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=200)

print "\n"+"*"*45
print "Author: 100"
print "*"*45+"\n"
main("SVM", AUtrain_Xn100, AUtrain_y100, AUtest_Xn100, AUtest_y100,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=200)
main("DTC", AUtrain_Xn100, AUtrain_y100, AUtest_Xn100, AUtest_y100,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=200)
main("KNN", AUtrain_Xn100, AUtrain_y100, AUtest_Xn100, AUtest_y100,
	[["F1"],["F2"],["F3"],["F1","F2"],["F1","F3"],["F2","F3"],["all"]], tree_select=200)
