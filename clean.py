import pickle
import numpy as np
import featuremap
from featuremap import featuremap, metadatamap
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

meta = pickle.load( open( "metadata.p", "rb" ) )
data = pickle.load( open( "dataset.p", "rb" ) )

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


"""
This function takes a dataset and a max accepted number of standard deviations.
It only looks at non-fraction features, as the variance is large among fraction features 
If it finds a value that is further away from the mean than numberofstdev standard deviations, 
it prints the doc id, featurename and the value for visual inspection
"""
def outlierdetection(dataset, metadata, numberofstdev):
	for i in xrange(len(dataset[0])): #for every lexical and wordbased feature
		mean = np.mean(dataset[:, i])
		stdev = np.std(dataset[:, i])
		for j in xrange(len(dataset)):
			if 'frac' in featuremap[i]: #if it's a frequency feature
				pass
			else:
				if dataset[j][i] < (mean - (stdev*numberofstdev)) or dataset[j][i] > (mean + (stdev*numberofstdev)):
					print "Feature: %s Mean: %.4f(%.4f), Value: %.4f , document id: %s" %(featuremap[i], mean, stdev,  dataset[j][i], metadata[j][0])

#outlierdetection(data, meta, 5)

#i = featuremap.index('WB_num_words')

#----------------------------------------------------------------------------------------
#PCA
#----------------------------------------------------------------------------------------

def princomp(train):
	pca = PCA(copy=True)
	transformed = pca.fit_transform(train)
	components = pca.components_
	exp_variance = pca.explained_variance_ratio_

	x = np.array([i for i in range(len(exp_variance))])

	#print "Explained variance", exp_variance
	M = (train-np.mean(train.T,axis=1)).T # subtract the mean (along columns)
 	eigv, eigw = np.linalg.eig(np.cov(M)) 
	plt.plot(x, exp_variance)
	plt.title("Explained Variance")
	plt.ylabel("Exp. variance")
	plt.xlabel("Components")
	plt.show()

def princomp_transform(trainset, testset, components):
	#print trainset[0]
	pca = PCA(n_components=components, copy=True, whiten=False)

	pca.fit(trainset)
	X_train_trans = pca.transform(trainset)
	X_test_trans = pca.transform(testset)
	#orig = pca.inverse_transform(X_train_trans)
	#print orig[0]
	return X_train_trans, X_test_trans

"""
This function returns True if the passed parameter is an integer
def is_int(var):
	try:
    		var= int(var)
	except ValueError:
    		pass  # it was a string, not an int.

"""