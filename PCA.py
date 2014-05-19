from __future__ import division
from sklearn.cluster import KMeans
import pylab as plt
import numpy as np
from sklearn.decomposition import PCA


def pca(no_pc, data):

	# Calculating eigenvalues and eigenvectors
	pca_data = data.T
	cov = np.cov(pca_data)
	w, v = np.linalg.eig(cov)

	# Projection on the two first principal components
	P = v.T
	reduced_data = np.dot(P[:no_pc],pca_data).T

	return reduced_data


def princomp2(dataset):
	print "#" * 45
	print "PCA"
	print "#" * 45
	pca = PCA(copy=True)
	transformed = pca.fit_transform(dataset)
	components = pca.components_
	exp_variance = pca.explained_variance_ratio_

	print "Explained variance", exp_variance
	M = (dataset-np.mean(dataset.T,axis=1)).T # subtract the mean (along columns)
 	eigv, eigw = np.linalg.eig(np.cov(M)) 

 	x = [1,2,3,4,5,6,7,8,9,10]

	plt.plot(x, eigv)
	plt.title("Eigenspectrum")
	plt.ylabel("Eigenvalue")
	plt.xlabel("Components")
	plt.show()


"""
This function visualizes the clusters of the dataset reduced to 2 components and plots it. 
It expects a dataset and optionally number of k for the clustering
"""
def princomp(dataset, k):
	pca2 = PCA(n_components=2)
	pca2.fit(dataset)
	transformed = pca2.transform(dataset)
	"""

	init = np.array([dataset[0], dataset[1]])
	

	kmeans = KMeans(n_clusters=k, copy_x=True)
	kmeans.fit(dataset) 
	clustermeans = kmeans.cluster_centers_#getting 338D clustermeans from the normalized dataset

	print "#" * 45
	print "K-means clustering"
	print "#" * 45
	print "k = ", k
	print "Means of clusters before PCA transform:", clustermeans

	transformed_mean1 = pca2.transform(clustermeans[0])
	transformed_mean2 = pca2.transform(clustermeans[1])	

	meanx = [transformed_mean1[0][0], transformed_mean2[0][0]]
	meany = [transformed_mean1[0][1], transformed_mean2[0][1]]
	"""
	return transformed


"""
def plotclass(transformed, meanx, meany, labels, k, task):
	plt.title("%s: Projected on 2 PC with clustermeans" %(task))
	plt.xlabel("first principal component")
	plt.ylabel("second principal component")

	for j in xrange(k):
		sortedlabels = sorted(set(labels))
		plt.plot([transformed[i][0] for i in xrange(len(transformed)) if labels[i] == sortedlabels[j]], [transformed[i][1] for i in xrange(len(transformed)) if labels[i] == sortedlabels[j]], 'x', label = sortedlabels[j])

	plt.plot(meanx, meany, 'ro', label = "Cluster means")
	plt.legend(loc='best')
	plt.show()

"""

