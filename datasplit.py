from __future__ import division
from collections import Counter
from featuremap import featuremap, metadatamap
import random
import pickle
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
import numpy as np



def natlan(metadata, data):

	random.seed(448)
	labels = np.array([1 if l == 'English' else 0 for l in metadata[:,-4]]) # 1 = Native (English), 0 = Non-native
	zipped = zip(labels,data)
	random.shuffle(zipped)
	split = int(len(zipped)*0.80)
	train, test = zip(*zipped[:split]), zip(*zipped[split:])

	print "Native Language" 
	print "distribution in train set"
	counted = Counter(train[0])

	print "Native Language: English", counted[1]
	print "Native Language: Not English", counted[0]
	print ""
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

	print "Grade" 
	print "Distribution in train set"
	counted = Counter(train[0])

	print "Grade: D", counted[0]
	print "Grade: M", counted[1]
	print ""
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

	print "Grade" 
	print "Distribution in train set"
	counted = Counter(train[0])

	print "Level 1", counted[1]
	print "Level 2", counted[2]
	print "Level 3", counted[3]
	print "Level 4", counted[4]

	print ""
	print "Train set size ", len(train[0])
	print "Test set size ", len(test[0])
	print "-" *45

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


"""
Select a featureset and ignore all the other features: 'LEX' for lexical features, 'WB' for word-based features, 'SYN' for syntactic features, 'STRUC' for structural features. 
It returns a new train and test set consisting only of the selected features
"""
def select_featureset(trainset, testset, featureset):
	trainset = np.copy(trainset)
	testset = np.copy(testset)

	#Indices for the different features
	LEX_start = 0
	LEX_end = featuremap.index('LEX_frac_)')

	WB_start = featuremap.index('WB_num_words')
	WB_end = featuremap.index('WB_frac_word_len20')

	SYN_start = featuremap.index('SYN_frac_,')
	SYN_end = featuremap.index('SYN_frac_POS_X')

	STRUC_start = featuremap.index('STRUC_num_sent')
	STRUC_end = len(featuremap)

	if featureset == 'LEX':
		train_LEX = trainset[LEX_start:LEX_end+1,:]
		test_LEX = testset[LEX_start:LEX_end+1,:]

		return train_LEX, test_LEX

	elif featureset == 'WB':
		train_WB = trainset[WB_start:WB_end+1,:]
		test_WB = testset[WB_start:WB_end+1,:]

		return train_WB, test_WB

	elif featureset == 'SYN':
		train_SYN = trainset[SYN_start:SYN_end+1,:]
		test_SYN = testset[SYN_start:SYN_end+1,:]

		return train_SYN, test_SYN

	elif featureset == 'STRUC':
		train_STRUC = trainset[STRUC_start:STRUC_end+1,:]
		test_STRUC = testset[STRUC_start:STRUC_end+1,:]

		return train_STRUC, test_STRUC

	else:
		print 'Unknown featureset'

def sub(subset):
	LEX_start = 0
	LEX_end = featuremap.index('LEX_frac_)')

	WB_start = featuremap.index('WB_num_words')
	WB_end = featuremap.index('WB_frac_word_len20')

	SYN_start = featuremap.index('SYN_frac_,')
	SYN_end = featuremap.index('SYN_frac_POS_X')

	STRUC_start = featuremap.index('STRUC_num_sent')
	STRUC_end = len(featuremap)

	if subset == 'LEX':
		return np.arange(LEX_start, LEX_end+1)

	elif subset == 'WB':
		return np.arange(WB_start, WB_end+1)
	elif subset == 'SYN':
		return np.arange(SYN_start, SYN_end+1)
	elif subset == 'STRUC':
		return np.arange(STRUC_start, STRUC_end)
	else:
		print "Unknown featureset"