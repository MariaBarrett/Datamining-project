from __future__ import division 
import numpy as np
import featuremap

dataset = np.array([[100,3,3,4],[2,3,5,6],[3,4,4,5]])

mean = np.mean(dataset, axis=0)
stdev = np.std(dataset, axis=0)

print mean
print stdev

def outlierdetection(dataset):
	for i in xrange(len(dataset[0])): #for every feature
		for datapoint in dataset:
			if datapoint[i] < (mean[i] - (stdev[i])) or datapoint[i] > (mean[i] + (stdev[i])):
				print featuremap.featuremap[i]
				print datapoint[i]

outlierdetection(dataset)