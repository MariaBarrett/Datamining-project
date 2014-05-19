from __future__ import division
from sklearn.cluster import KMeans
import pylab as plt
import numpy as np
from sklearn.decomposition import PCA

"""
This function visualizes the clusters of the dataset reduced to 2 components and plots it. 
It expects a dataset and optionally number of k for the clustering
"""
def cluster(dataset, labels, k):
	#pca2 = PCA(n_components=2)
	#pca2.fit(dataset)
	#transformed = pca2.transform(dataset)
	#init = np.array([dataset[0], dataset[1]])
	kmeans = KMeans(n_clusters=k, copy_x=True)
	kmeans.fit(dataset,labels) 
	clustermeans = kmeans.cluster_centers_#getting 338D clustermeans from the normalized dataset
	return clustermeans

def largestvariance(means):
	means = np.mean(means, axis=0)
	stdev = np.std(means, axis=0)

