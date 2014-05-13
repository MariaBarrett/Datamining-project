from __future__ import division
from collections import Counter
from featuremap import featuremap, metadatamap
import random
import pickle
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
import numpy as np

metadata = pickle.load( open( "metadata.p", "rb" ) )
labels = np.copy(metadata[:,-4])
data = pickle.load( open( "dataset.p", "rb" ) )

def natlan(metadata, data):

	random.seed(448)
	labels = np.array([1 if l == 'English' else 0 for l in metadata[:,-4]]) # 1 = Native (English), 0 = Non-native
	zipped = zip(labels,data)
	random.shuffle(zipped)
	split = int(len(zipped)*0.80)
	train, test = zip(*zipped[:split]), zip(*zipped[split:])

	return train, test

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

	return train, test

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

	return train, test



natlan_train, natlan_test = natlan(metadata,data)
grade_train, grade_test = grade(metadata,data)
level_train, level_test = level(metadata,data)