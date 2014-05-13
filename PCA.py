from __future__ import division
from sklearn.cluster import KMeans
import clean
import datasplit
import matplotlib.pyplot as plt
import numpy as np


def pca(no_pc, data):

	# Calculating eigenvalues and eigenvectors
	pca_data = data.T
	cov = np.cov(pca_data)
	w, v = np.linalg.eig(cov)

	# Projection on the two first principal components
	P = v.T
	reduced_data = np.dot(P[:no_pc],pca_data).T

	return reduced_data


