from __future__ import division 
import numpy as np
import featuremap

dataset = np.array([[100,3,3,4],[2,3,5,6],[3,4,4,5]])

meta = np.array([[2,3],[2,3],[3,2]])

mean = np.mean(dataset, axis=0)
stdev = np.std(dataset, axis=0)

"""
This function takes a dataset and a max acceptec number of standard deviations.
If it finds a value that is further away from the mean than numberofstdev standard deviations, 
it prints the featurename and the value for visual inspection
"""
def outlierdetection(dataset, numberofstdev):
	for i in xrange(len(dataset[0])): #for every feature
		for datapoint in dataset:
			if datapoint[i] < (mean[i] - (stdev[i]*numberofstdev)) or datapoint[i] > (mean[i] + (stdev[i]*numberofstdev)):
				print featuremap.featuremap[i]
				print datapoint[i]

#outlierdetection(dataset)

#remove if there are more than one author

"""
This function returns the indexname of a feature when given the featurename and the list (either featuremap.metadatamap or featuremap.featuremap)
"""
def get_featurenumber(feature, list='featuremap.featuremap'):
	for i in xrange(len(list)):
		if featuremap.metadatamap[i] == featurename:
			return i

"""
This function removes datapoints if a feature of a certain index has a value that is either min, max or equal to a specified value. 
To activate either minval, maxval or equal, e.g. write minval=1 and it will clean all datapoints in both metadata and dataset with values above 1.0 at the specified index 
c counts how many datapoints have been deleted in each set. 
"""

def clean_from_meta(metadata, dataset, index, minval=None, maxval=None, equal=None):
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

#number of authors > 1

"""
This function count number of different target groups
It expects the metadata arrays and the dataset and name of the desired metadata feature name
"""
def countgroups(metadata, featurename):
	index = get_featurenumber(featurename, featuremap.metadata)
	different_values={}
	for datapoint in metadata:
		if datapoint[i] in different_values:
			different_values[datapoint[i]] +=1
		else: 
			different_values[datapoint] = 1
	return different_values


