'''
Implement Keselj, Vlado, et al.'s n-gram author attribution on BAWE
This file calculates distances and uses knn to find authors from ngram profiles
profiles are loaded via pickle. To generate profiles run ngram.py
'''

import metadata, nltk, pickle, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

data = metadata.data

def normalize(ds):
	ds_norm = (ds - ds.mean()) / (ds.max() - ds.min())
	return ds_norm

def dissimilarity(profile1, profile2):
	''' Implements the dissimilarity measure (algorithm 1, 'Algorithm 1 Profile Dissimilarity(profile1, profile2)')
	from Keselj, Vlado, et al. '''

	sigma = 0 #sigma is the summation variable. It's final value will be the dissimilarity score

	# n-grams x contained in profile1 or profile2
	X = list(profile1['ngram']) + list(profile2['ngram'])
	X = list(set(X))

	for x in X:
		# let f1 and f2 be frequencies of x in profile1 and profile2 (zero if they are not included)
		if any(profile1['ngram'] == x) == True:
			f1 = float(profile1[profile1['ngram'] ==x].freq)
		else:
			f1 = 0
		if any(profile2['ngram'] == x) == True:
			f2 = float(profile2[profile2['ngram'] ==x].freq)
		else:
			f2 = 0
		# add square of normalized differences
		sigma += (2 * (f1-f2)/(f1+f2))**2
	return sigma

def get_author(did):
	''' takes a document id and returns the associated author
	'''
	did = str(did) # sanitize input
	global data
	return int(data[data.id == did].student_id)

def get_likely_author(aids):
	''' takes a list of author ids and returns the most common value
	'''
	c = Counter(aids)
	return c.items()[0][0]



# Load data
print "Loading data"
start = time.clock()
frequencies = pickle.load( open('ng_frequencies.p', 'rb'))
doc_id = pickle.load( open('ng_doc_id.p', 'rb'))
doc_freq = pickle.load( open('ng_doc_freq.p', 'rb'))
authors = pickle.load( open('ng_authors.p', 'rb'))
profiles = pickle.load( open('ng_profiles.p', 'rb'))
print "Loading complete in ", time.clock() - start

L = 1000

N = 3 # number of nearest neighbours to vote for author

# 3 experiments
# distances using top overall frequencies
#	Measure distances between author profiles and other documents
# distances using ALL frequencies

# measure distances using ALL ngrams

nearest_author = []
author_by_vote = []

for i, d in enumerate(doc_freq):
	d = doc_freq[0]
	dissimilarities = []
	for p in profiles:
		dissimilarities.append(dissimilarity(d, p))
	index = np.argsort(dissimilarities) #sorts ascending (1,2,3...)

	nearest_author.append(get_author(doc_id[index[0]]))
	author_by_vote.append(get_likely_author(doc_id[index[:N]]))

pickle.dump( nearest_author, open('nearest_author.p', 'wb'))
pickle.dump( author_by_vote, open('author_by_vote.p', 'wb'))












