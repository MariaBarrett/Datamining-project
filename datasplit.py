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

natlan_train, natlan_test = natlan(metadata,data)