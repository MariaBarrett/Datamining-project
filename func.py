from __future__ import division
from collections import Counter
from featuremap import featuremap, metadatamap
import random
import pickle
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import pylab as pl

def natlan(metadata, data):

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


def sub(subset, best=None):
	
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
	if len(indices) > 0:
		return indices
	else:
		return subset[0]


def tree_selection(train_data,train_labels,number_of_features):
  
  forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
  forest.fit(train_data, train_labels)
  importances = forest.feature_importances_
  indices = np.argsort(importances)[::-1]

  return indices[:number_of_features]


def inspect_tree_selection(train_data,train_labels, task):
  
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
	"""This function normalizes a dataset. It is possible to include
	a test dataset to be normalized with the mean and std calculated from
	the training dataset."""

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

def inspect_pca(train):
	
	pca = PCA(copy=True)
	transformed = pca.fit_transform(train)
	components = pca.components_
	exp_variance = pca.explained_variance_ratio_

	x = np.array([i for i in range(len(exp_variance))])

	#print "Explained variance", exp_variance
	plt.plot(x, exp_variance)
	plt.title("Explained Variance")
	plt.ylabel("Exp. variance")
	plt.xlabel("Components")
	plt.show()

def pca_transform(trainset, testset, components):

	pca = PCA(n_components=components, copy=True, whiten=False)
	pca.fit(trainset)
	X_train_trans = pca.transform(trainset)
	X_test_trans = pca.transform(testset)

	return X_train_trans, X_test_trans





