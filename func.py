from __future__ import division
from collections import Counter
from featuremap import featuremap, metadatamap
import random
import pickle
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
import pylab as pl

def natlan(metadata, data):
	""" Makes an 80-20 train-test split for the native language task.
	Before splitting the dataset is shuffled. """

	random.seed(448)
	labels = np.array([1 if l == 'English' else 0 for l in metadata[:,-4]]) # 1 = Native (English), 0 = Non-native
	zipped = zip(labels,data)
	random.shuffle(zipped)
	split = int(len(zipped)*0.80)
	train, test = zip(*zipped[:split]), zip(*zipped[split:])

	print "Native Language:" 
	print "Train set size ", len(train[0])
	print "Test set size ", len(test[0])
	print "-" *45

	return np.array(train[0]), np.array(train[1]), np.array(test[0]), np.array(test[1])

def grade(metadata, data):
	""" Makes an 80-20 train-test split for the grade task.
	Before splitting all texts with the grade value 'unknown' are removed
	from dataset and the dataset is shuffled. """

	random.seed(448)
	unknown = np.where(metadata[:,8]=="unknown")[0]
	data_clean = np.delete(data, unknown, 0)
	metadata_clean = np.delete(metadata, unknown, 0)
	grade_map = {'M':1, 'D':0}
	labels = np.array([grade_map[g] for g in metadata_clean[:,8]]) # 1 = M, 0 = D
	zipped = zip(labels,data_clean)
	random.shuffle(zipped)
	split = int(len(zipped)*0.80)
	train, test = zip(*zipped[:split]), zip(*zipped[split:])

	print "Grade:"
	print "Train set size ", len(train[0])
	print "Test set size ", len(test[0])
	print "-" *45
	
	return np.array(train[0]), np.array(train[1]), np.array(test[0]), np.array(test[1])

def level(metadata, data):
	""" Makes an 80-20 train-test split for the academic level task.
	Before splitting all texts with the level value 'unknown' are removed
	from dataset and the dataset is shuffled. """

	random.seed(448)
	unknown = np.where(metadata[:,1]=="unknown")[0]
	data_clean = np.delete(data, unknown, 0)
	metadata_clean = np.delete(metadata, unknown, 0)
	
	labels = np.array([int(l) for l in metadata_clean[:,1]]) # 1, 2, 3, 4
	
	zipped = zip(labels,data_clean)
	random.shuffle(zipped)
	split = int(len(zipped)*0.80)
	train, test = zip(*zipped[:split]), zip(*zipped[split:])

	print "Academic level:" 
	print "Train set size ", len(train[0])
	print "Test set size ", len(test[0])
	print "-" *45

	return np.array(train[0]), np.array(train[1]), np.array(test[0]), np.array(test[1])

def author(metadata, data, min_text, no_authors, in_test=1, feat_sort=False):
	""" Makes a train-test split for the author task. The parameter min_text
	defines the number of texts each author must have contributed to the dataset
	in order to be included in the sample. The parameter in_test defines how many
	texts from each author should be alocated to the test data. These texts are
	randomly selected from all of the authors texts. The feat_sort parameter takes
	a tuple with a column argument and a value argument in order to exclude all data
	points not containing the given value in the given column. """

	if feat_sort:
		sort_delete = np.where(metadata[:,feat_sort[0]] != feat_sort[1])[0]
		data = np.delete(data, sort_delete, 0)
		metadata = np.delete(metadata, sort_delete, 0)

	random.seed(448)
	author_count = Counter(metadata[:,-1])
	delete_authors = np.array([a for a in author_count.keys() if author_count[a] < min_text])
	delete_index = np.array([i for a in delete_authors for i in np.where(metadata[:,-1]==a)[0]])
	data_clean = np.delete(data, delete_index, 0)
	metadata_clean = np.delete(metadata, delete_index, 0)

	most_freq = [(v, k) for k, v in author_count.items()]; most_freq.sort(); most_freq.reverse()
	mf_authors = [m[1] for m in most_freq][:no_authors]
	mf_index = np.array([i for m in mf_authors for i in np.where(metadata_clean[:,-1]==m)[0]])
	
	labels = np.array([int(metadata_clean[:,-1][i]) for i in mf_index])
	data_reduced = np.array([data_clean[i] for i in mf_index])

	test_index = np.array([i for l in set(labels) for i in random.sample(np.where(labels==l)[0],in_test)])
	zipped = zip(labels,data_reduced)
	test = zip(*[zipped[ti] for ti in test_index])
	train = zip(*[i for j, i in enumerate(zipped) if j not in test_index])

	print "Author:" 
	print "Train set size ", len(train[0])
	print "Test set size ", len(test[0])
	print "Text per author in test set", in_test
	print "Text per author in train set", str(most_freq[:no_authors][-1][0])+"-"+str(most_freq[:no_authors][0][0])
	print "-" *45

	return np.array(train[0]), np.array(train[1]), np.array(test[0]), np.array(test[1])


def sub(subset):
	""" Given one or more subset names (F1, F2, F3 or all) in a list,
	this function returns the indices needed to find the specific
	feature combinations. """
	
	LEX_start = 0
	LEX_end = featuremap.index('WB_frac_word_len20')

	SYN_start = featuremap.index('SYN_frac_,')
	SYN_end = featuremap.index('SYN_frac_POS_X')

	STRUC_start = featuremap.index('STRUC_num_sent')
	STRUC_end = len(featuremap)

	indices = list()

	if 'F1' in subset:
		indices += range(LEX_start, LEX_end+1)
	if 'F2' in subset:
		indices += range(SYN_start, SYN_end+1)
	if 'F3' in subset:
		indices += range(STRUC_start, STRUC_end)
	if 'all' in subset:
		indices += range(LEX_start, STRUC_end)

	return indices


def tree_selection(train_data,train_labels,number_of_features):
	""" Returns the indices for the best parameters of a given dataset
	and it's target labels. The number_of_features parameter should be
	choosen by visual inspection using the inspect_tree_selection function. """ 

	forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
	forest.fit(train_data, train_labels)
	importances = forest.feature_importances_
	indices = np.argsort(importances)[::-1]

	return indices[:number_of_features]


def inspect_tree_selection(train_data,train_labels, task):
	""" Given a dataset and it's target labels, this
	function sorts the best features and prints and visualize them """

	forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
	forest.fit(train_data, train_labels)
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print "-"*45
	print("\nFeature ranking for %s task:" %(task))

	for f in range(len(indices)):
	  print("%d. feature, name: %s, importance: %f" % (f + 1, featuremap[indices[f]], importances[indices[f]]))

	# Plot the feature importances of the forest
	pl.figure()
	n = train_data.shape[1]
	pl.title("%s: Sorted feature importance" %(task))
	pl.bar(range(n), importances[indices][:n], color="black", align="center")
	pl.xlim([-1, (n)])
	pl.show()

def normalize(traindata,testdata):
	""" This function normalizes a dataset (0 mean, unit variance) given a dataset that has
	already been splitted into separate train and test sets. """

	mean = np.mean(traindata, axis=0)
	std = np.std(traindata, axis=0)

	traindata_normalized = np.copy((traindata-mean)/std)
	testdata_normalized = np.copy((testdata-mean)/std)

	train_nan = np.isnan(traindata_normalized)
	test_nan = np.isnan(testdata_normalized)
	train_inf = np.isinf(traindata_normalized)
	test_inf = np.isinf(testdata_normalized)

	traindata_normalized[train_nan] = 0
	testdata_normalized[test_nan] = 0
	traindata_normalized[train_inf] = 0
	testdata_normalized[test_inf] = 0

	return traindata_normalized, testdata_normalized

def inspect_pca(train, expvar_threshold=0.95):
	""" Given a dataset and a threshold for the explained variance, this
	function returns the number of principal components to use in order to
	account for the given amount of explained variance and plots the 
	explained variance for all of the the new features. """

	pca = PCA(copy=True)
	transformed = pca.fit_transform(train)
	components = pca.components_
	exp_variance = pca.explained_variance_ratio_

	c = 0
	for i, ev in enumerate(exp_variance):
		c += ev
		if c > expvar_threshold:
			expvar_index = i
			break

	x = np.array([i for i in range(len(exp_variance))])

	#print "Explained variance", exp_variance
	plt.plot(x, exp_variance)
	plt.title("Explained Variance")
	plt.ylabel("Exp. variance")
	plt.xlabel("Components")
	plt.show()

	return expvar_index

def pca_transform(trainset, testset, components):
	""" Transform a training dataset and a test dataset by mirroring the
	data on the specified number of principal components for the training
	dataset. """

	pca = PCA(n_components=components, copy=True, whiten=False)
	pca.fit(trainset)
	X_train_trans = pca.transform(trainset)
	X_test_trans = pca.transform(testset)

	return X_train_trans, X_test_trans





