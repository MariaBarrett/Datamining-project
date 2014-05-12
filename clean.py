import pickle
import numpy as np
import featuremap
from featuremap import featuremap

metadata = pickle.load( open( "metadata.p", "rb" ) )
labels = np.array([a_id[:4] for a_id in metadata[:,0]])
traindata = pickle.load( open( "dataset.p", "rb" ) )

def normalize(traindata,testdata=False):
	"""This function normalizes a dataset. It is possible to include
	a test dataset to be normalized with the mean and std calculated from
	the training dataset."""

	mean = np.mean(traindata, axis=0)
	std = np.std(traindata, axis=0)

	# Remove features with standard deviation = 0, as they are not interesting
	zero_feat = np.where(std==0)[0]

	if len(zero_feat) > 0:
		traindata = np.delete(traindata, zero_feat, 1)
		mean = np.delete(mean, zero_feat)
		std = np.delete(std, zero_feat)
		if testdata != False:
			testdata = np.delete(testdata, zero_feat, 1)

	# Normalize data
	traindata_normalized = np.copy((traindata-mean)/std)
	
	if testdata != False:
			testdata_normalized = np.copy((testdata-mean)/std)

			return traindata_normalized, testdata_normalized

	return traindata_normalized

def lasses(labels,data,print_l=False):
	"""This function calculates an Lasse's K for feature selection. The
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
	l_names = zip(l_scores, featuremap)
	l_names.sort(key = lambda t: t[0])

	if print_l != False:
		for i in range(print_l):
			print str(i+1)+". "+l_names[i][1]

	return l_names


def outlierdetection(dataset, numberofstdev):
	"""This function takes a dataset and a max acceptec number of
	standard deviations. If it finds a value that is further away
	from the mean than numberofstdev standard deviations, it prints
	the featurename and the value for visual inspection"""

	for i in xrange(len(dataset[0])): #for every feature
		for datapoint in dataset:
			if datapoint[i] < (mean[i] - (stdev[i]*numberofstdev)) or datapoint[i] > (mean[i] + (stdev[i]*numberofstdev)):
				print featuremap.featuremap[i]
				print datapoint[i]

#outlierdetection(dataset)

#remove if there are more than one author


def get_featurenumber(feature, list='featuremap.featuremap'):
	"""This function returns the indexname of a feature when given
	the featurename and the list (either featuremap.metadatamap or
	featuremap.featuremap)"""

	for i in xrange(len(list)):
		if featuremap.metadatamap[i] == featurename:
			return i


def clean_from_meta(metadata, dataset, index, minval=None, maxval=None, equal=None):
	"""This function removes datapoints if a feature of a certain index has a value that
	is either min, max or equal to a specified value. To activate either minval, maxval or
	equal, e.g. write minval=1 and it will clean all datapoints in both metadata and dataset
	with values above 1.0 at the specified index c counts how many datapoints have been
	deleted in each set."""

	c=0
	for i in xrange(len(metadata)): #for every datapoint
		if minval:
			if metadata[i][index] < float(minval):
				c+=1
				metadata = np.delete(metadata, i, axis=0)  
				dataset = np.delete(dataset, i, axis=0)  
		if maxval:
			if metadata[i][index] > float(maxval):
				c+=1
				metadata = np.delete(metadata, i, axis=0)  
				dataset = np.delete(dataset, i, axis=0)  
		if equal:
			if metadata[i][index] == float(equal):
				c+=1
				metadata = np.delete(metadata, i, axis=0)  
				dataset = np.delete(dataset, i, axis=0)  
	print "For index %s, %s datapoint(s) were deleted in each dataset" %(index, c)
	return metadata, dataset, c

newmeta, newdata, deleted = clean_from_meta(meta, dataset, 1, equal=2)

print newmeta, newdata
#number of authors > 1


def countgroups(metadata, featurename):
	"""This function count number of different target groups.
	It expects the metadata arrays and the dataset and name of
	the desired metadata feature name"""

	index = get_featurenumber(featurename, featuremap.metadata)
	different_values={}
	for datapoint in metadata:
		if datapoint[i] in different_values:
			different_values[datapoint[i]] +=1
		else: 
			different_values[datapoint] = 1
	return different_values