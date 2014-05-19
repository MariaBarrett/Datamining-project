'''
Implement Keselj, Vlado, et al.'s n-gram author attribution on BAWE
This file calculates distances and uses knn to find authors from ngram profiles
profiles are loaded via pickle. To generate profiles run ngram.py
'''

import metadata, nltk, pickle, time, clean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter


data = metadata.data
n = 3
profile_size = 2000





# Load data

print "Loading data"
start = time.clock()


train = pickle.load( open('ng_train'+str(n)+'.p', 'rb'))
train_t = pickle.load( open('ng_train_t'+str(n)+'.p', 'rb'))
train_freq = pickle.load( open('ng_train_freq'+str(n)+'.p', 'rb'))
train_ngram = pickle.load( open('ng_train_ngram'+str(n)+'.p', 'rb'))

test = pickle.load( open('ng_test'+str(n)+'.p', 'rb'))
test_t = pickle.load( open('ng_test_t'+str(n)+'.p', 'rb'))
test_ngram = pickle.load( open('ng_test_ngram'+str(n)+'.p', 'rb'))
test_freq = pickle.load( open('ng_test_freq'+str(n)+'.p', 'rb'))


'''
doc_id = pickle.load(open('ng_doc_id'+str(n)+'.p', 'rb'))
doc_ngram = pickle.load(open('ng_doc_ngram'+str(n)+'.p', 'rb'))
doc_freq = pickle.load(open('ng_doc_freq'+str(n)+'.p', 'rb'))

authors = pickle.load(open('ng_authors'+str(n)+'.p', 'rb'))
auth_ngram = pickle.load(open('ng_auth_ngram'+str(n)+'.p', 'rb'))
auth_freq = pickle.load(open('ng_auth_freq'+str(n)+'.p', 'rb'))
'''


print "Loading complete in ", time.clock() - start

#N = 3 # number of nearest neighbours to vote for author

# 3 experiments
# distances using top overall frequencies
#	Measure distances between author profiles and other documents
# distances using ALL frequencies

# measure distances using ALL ngrams

def normalize(data):
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)

	normalized = (data-mean)/std

	return normalized


def dissimilarity(ngram1, freq1, ngram2, freq2, ):
	''' Implements the dissimilarity measure (algorithm 1, 'Algorithm 1 Profile Dissimilarity(freq1, freq2)')
	from Keselj, Vlado, et al. '''
	global profile_size

	def to_float(a):
		'''converts a single length array to float. return 0 if len != 1'''
		if any(a):
			return float(a)
		else:
			return 0

	def shorten_profile(profile):
		''' takes the profile_size amount of elements of input and returns. if profile_size is larger than amounts of elements, return profile and avoid raising out of index error
		'''
		global profile_size

		if len(profile) > profile_size:
			return profile[:profile_size]
		else:
			return profile

	ngram1 = shorten_profile(ngram1)
	freq1 = normalize(shorten_profile(freq1))
	ngram2 = shorten_profile(ngram2)
	freq2 = normalize(shorten_profile(freq2))

	sigma = 0 #sigma is the summation variable. It's final value will be the dissimilarity score

	# n-grams x contained in freq1 or freq2
	X = list(ngram1) + list(ngram2)
	X = list(set(X))

	for x in X:
		# let f1 and f2 be frequencies of x in freq1 and freq2 (zero if they are not included)
		f1 = to_float(freq1[ngram1 == x])
		f2 = to_float(freq2[ngram2 == x])
		# add square of normalized differences
		sigma += (2 * (f1-f2)/(f1+f2))**2
	return sigma



print "Testing L1"
predictions = []

for i in range(len(test)):
	dissimilarities = []
	for j in range(len(train_freq)):
		dissimilarities.append(dissimilarity(test_ngram[i], test_freq[i], train_ngram[j], train_freq[j]))
	index = np.argsort(dissimilarities)
	pred = index[0]
	predictions.append(pred)
	print test[i], train_t[i] == pred, index, train_t[i], dissimilarities
'''


doc_predict = []

for df, dn, did, i in zip(doc_freq, doc_ngram, doc_id, range(len(doc_id))):
	#df = doc_freq[3]; dn = doc_ngram[3] #for testing purpose only!!!
	dissimilarities = []
	for af, an in zip(auth_freq, auth_ngram):
		dissimilarities.append(dissimilarity(dn, df, an, af))
	index = np.argsort(dissimilarities) #sorts ascending (1,2,3...)
	predicton = authors[index[0]]
	doc_predict.append(predicton)
	print did, get_author_from_id(did) == predicton, index, dissimilarities[:4]


def predict_single_doc(i):
	global doc_freq, doc_ngram, auth_freq, auth_ngram

	df = doc_freq[i]; dn = doc_ngram[i]
	dissimilarities = []
	for af, an in zip(auth_freq, auth_ngram):
		dissimilarities.append(dissimilarity(dn, df, an, af))
	index = np.argsort(dissimilarities) #sorts ascending (1,2,3...)
	return dissimilarities, index



def check_top_x_likely_author(number_of_docs, random=True):
	global doc_id


	if random == True:
		this_range = range(len(doc_id))
		random.shuffle(this_range)
		this_range = this_range[:number_of_docs]
	else:
		this_range = range(number_of_docs)

	# Get predictions
	index = {}
	for i in this_range:
		index[i] = predict_single_doc(i)

	# Display how many authors are predicted as being more likely correct than current author
	for i in this_range:
		a = int(doc_id[i][:-1])
		print doc_id[i], where(index[i]==1)

	return index


'''






