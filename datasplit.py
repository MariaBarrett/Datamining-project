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

metadata = pickle.load( open( "metadata.p", "rb" ) )
data = pickle.load( open( "dataset.p", "rb" ) )
plusone = np.where(metadata[:,9] != "1")[0] # Get indexes for all texts written by more than one person
data, metadata = np.delete(data, plusone, 0), np.delete(metadata, plusone, 0) # Remove all texts written by more than one person from data

def natlan(metadata, data):

	random.seed(448)
	labels = np.array([1 if l == 'English' else 0 for l in metadata[:,-4]]) # 1 = Native (English), 0 = Non-native
	zipped = zip(labels,data)
	random.shuffle(zipped)
	split = int(len(zipped)*0.80)
	train, test = zip(*zipped[:split]), zip(*zipped[split:])

	"""
	print "Native Language" 
	print "distribution in train set"
	counted = Counter(train[0])

	print "Native Language: English", counted[1]
	print "Native Language: Not English", counted[0]
	print "Choosing most numerous class - baseline:", max([counted[0], counted[1]]) / (counted[0]+counted[1])
	print ""
	print "Train set size ", len(train[0])
	print "Test set size ", len(test[0])
	print "-" *45
	"""

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

	"""
	print "Grade" 
	print "Distribution in train set"
	counted = Counter(train[0])

	print "Grade: D", counted[0]
	print "Grade: M", counted[1]
	print "Choosing most numerous class - baseline:", max([counted[0], counted[1]]) / (counted[0]+counted[1])
	print ""
	print "Train set size ", len(train[0])
	print "Test set size ", len(test[0])
	print "-" *45
	"""
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

	"""
	print "Academic level" 
	print "Distribution in train set"
	counted = Counter(train[0])

	print "Level 1", counted[1]
	print "Level 2", counted[2]
	print "Level 3", counted[3]
	print "Level 4", counted[4]
	print "Choosing most numerous class - baseline:", max([counted[1], counted[2], counted[3], counted[4]]) / (counted[1]+counted[2]+counted[3]+counted[4])
	print ""
	print "Train set size ", len(train[0])
	print "Test set size ", len(test[0])
	print "-" *45
	"""

	return np.array(train[0]), np.array(train[1]), np.array(test[0]), np.array(test[1])

def author(metadata, data, min_text, no_authors, in_test=1):

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

	return np.array(train[0]), np.array(train[1]), np.array(test[0]), np.array(test[1])


def sub(subset, best=None):
	LEX_start = 0
	LEX_end = featuremap.index('WB_frac_word_len20')
	#LEX_end = featuremap.index('LEX_frac_)')
	#WB_start = featuremap.index('WB_num_words')
	#WB_end = featuremap.index('WB_frac_word_len20')

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


#dataset[:,sub("SYN")]


def tree_selection(train_data,train_labels,number_of_features):
  forest = ExtraTreesClassifier(n_estimators=250,
                                random_state=0)
  forest.fit(train_data, train_labels)
  importances = forest.feature_importances_
  indices = np.argsort(importances)[::-1]

  return indices[:number_of_features] #[a.argsort()[-10:]] #[:number_of_features]

def inspect_tree_selection(train_data,train_labels, task):
  forest = ExtraTreesClassifier(n_estimators=250,
                                random_state=0)
  forest.fit(train_data, train_labels)
  importances = forest.feature_importances_
  std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
  indices = np.argsort(importances)[::-1]

  # Print the feature ranking
  print("Feature ranking:")

  for f in range(len(indices)):
      print("%d. feature, name: %s, importance: %f" % (f + 1, featuremap[indices[f]], importances[indices[f]]))

  # Plot the feature importances of the forest
  pl.figure()
  n = train_data.shape[1]
  print n
  pl.title("%s: Importance of %s most important features" %(task, n))
  pl.bar(range(n), importances[indices][:n],
         color="r", yerr=std[indices][:n], align="center")
  #pl.xticks(n), indices[:n])
  pl.xlim([-1, (n)])
  pl.show()


def anova(data,labels,number_of_features):
	"""This function calculates a Lasse's K for feature selection. The
	lower the value, the more effective the feature is in describing a given
	class labels"""

	unique = set(labels)
	group_std = list()

	for l in unique:
		index = np.where(labels==l)[0]
		group = np.array([data[i] for i in index])
		group_std.append(np.std(group, axis=0))

	av_mean_std = np.mean(np.array(group_std), axis=0)
	overall_std = np.std(data, axis=0)

	l_scores = av_mean_std/overall_std
	indices = np.argsort(l_scores)

	return indices[:number_of_features]






