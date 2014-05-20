from __future__ import division
from collections import Counter
from featuremap import featuremap, metadatamap
import func
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
NLtrain_y, NLtrain_X, NLtest_y, NLtest_X = func.natlan(metadata,data)
GRtrain_y, GRtrain_X, GRtest_y, GRtest_X = func.grade(metadata,data)
LEtrain_y, LEtrain_X, LEtest_y, LEtest_X = func.level(metadata,data)
AUtrain_y20, AUtrain_X20, AUtest_y20, AUtest_X20 = func.author(metadata,data,4,20,2)
AUtrain_yAH, AUtrain_XAH, AUtest_yAH, AUtest_XAH = func.author(metadata,data,4,20,2,(7,"AH"))
AUtrain_y100, AUtrain_X100, AUtest_y100, AUtest_X100 = func.author(metadata,data,4,100,2)

# NORMALIZED DATA SETS
NLtrain_Xn, NLtest_Xn = func.normalize(NLtrain_X, NLtest_X)
GRtrain_Xn, GRtest_Xn = func.normalize(GRtrain_X, GRtest_X)
LEtrain_Xn, LEtest_Xn = func.normalize(LEtrain_X, LEtest_X)
AUtrain_Xn20, AUtest_Xn20 = func.normalize(AUtrain_X20, AUtest_X20)
AUtrain_XnAH, AUtest_XnAH = func.normalize(AUtrain_XAH, AUtest_XAH)
AUtrain_Xn100, AUtest_Xn100 = func.normalize(AUtrain_X100, AUtest_X100)

# TREE SELECTION FEATURES EVALUATION PLOTS
func.inspect_tree_selection(NLtrain_Xn, NLtrain_y, "Native language")
func.inspect_tree_selection(GRtrain_Xn, GRtrain_y, "Grade")
func.inspect_tree_selection(LEtrain_Xn, LEtrain_y, "Academic level")
func.inspect_tree_selection(AUtrain_Xn20, AUtrain_y20, "Author 20")
func.inspect_tree_selection(AUtrain_XnAH, AUtrain_yAH, "Author AH20")
func.inspect_tree_selection(AUtrain_Xn100, AUtrain_y100, "Author 100")

#----------------------------------------------------------------------------------------
# SVM MAIN FUNCTION
#----------------------------------------------------------------------------------------

def main(classifier, X_train, y_train, X_test, y_test, subsets, tree_select=False, PCA=False):

	""" This function takes a training dataset and it's target labels and train a classifier
	("SVM", "DTC" or "KNN") and test the classifier on the given test dataset and it's target
	labels. The subsets parameter is used to define multiple different subsets to train the
	classifier on. The tree_select parameter defines another subset, but utilizes a decision tree
	algorithm to find the best features. The parameter is set to the number of "best features" you
	want to use. The PCA parameter takes the number of principal components to mirror the dataset on
	and hence reduce the feature space size to. """
	
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


	for ss in subsets:

		print '-'*45
		print '+'.join(ss)+" features"
		print '-'*45+"\n"	

		if ss == ["TREE SELECTION"]:
			feat_index = func.tree_selection(X_train, y_train, tree_select)
		else: 
			feat_index = func.sub(ss)

		if PCA:
			X_train, X_test = func.pca_transform(X_train[:,feat_index], X_test[:,feat_index], PCA)
			feat_index = np.arange(PCA)
			print "Using PCA. No. of PC:", PCA
		
		clf.fit(X_train[:,feat_index], y_train)

		print "Best parameters: ", clf.best_params_
		print "\n0-1 loss", clf.score(X_test[:,feat_index], y_test)
		print "Baseline", stats.mode(y_test)[1][0]/len(y_test) # ZeroR baseline

		print "\nDetailed classification report:"
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


print "\n"+"*"*45
print "PCA results: explained variance = 95 %"
print "*"*45+"\n"
noPC_NL = func.inspect_pca(NLtrain_Xn, "Native language")
noPC_GR = func.inspect_pca(GRtrain_Xn, "Grade")
noPC_LE = func.inspect_pca(LEtrain_Xn, "Academic level")
noPC_AU20 = func.inspect_pca(AUtrain_Xn20, "Author 20")
noPC_AUAH = func.inspect_pca(AUtrain_XnAH, "Author AH20")
noPC_AU100 = func.inspect_pca(AUtrain_Xn100, "Author 100")

main("SVM", NLtrain_Xn, NLtrain_y, NLtest_Xn, NLtest_y,
	[["all"]], PCA=noPC_NL)
main("SVM", GRtrain_Xn, GRtrain_y, GRtest_Xn, GRtest_y,
	[["all"]], PCA=noPC_GR)
main("KNN", LEtrain_Xn, LEtrain_y, LEtest_Xn, LEtest_y,
	[["all"]], PCA=noPC_LE)
main("SVM", AUtrain_Xn20, AUtrain_y20, AUtest_Xn20, AUtest_y20,
	[["all"]], PCA=noPC_AU20)
main("SVM", AUtrain_XnAH, AUtrain_yAH, AUtest_XnAH, AUtest_yAH,
	[["all"]], PCA=noPC_AUAH)
main("SVM", AUtrain_Xn100, AUtrain_y100, AUtest_Xn100, AUtest_y100,
	[["all"]], PCA=noPC_AU100)

func.best_vc_pca()


