'''
Implement Keselj, Vlado, et al.'s n-gram author attribution on BAWE
'''

import metadata, pickle, nltk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

data = metadata.data

# test for n = [3,4,5]
n = 3
max_authors = 20

did = 'id' # variable with string representation of Document ID column
aid = 'student_id' # variable with string representation of Author ID column


train, test = metadata.split_authors_pd(data, max_authors=max_authors)

# Note that both train and test contains the same authors - the only authors that have enough documents to be part of the author split.
authors = train[aid].unique()

profiles = []

def get_ngrams(text, n):
	history = []
	length = len(text)
	for i in range(length-n):
		history.append(text[i:i+n])
	return history

def count_ngram(text, n=n):
	''' takes a string and return the number of ngrams and the count for each unique ngram in the string
	'''

	#ngrams = nltk.util.ngrams(text, n)
	ngrams = get_ngrams(text, n)

	number_of_ngrams = len(text)-(n-1)
	ngram_counter = Counter(ngrams)
	return ngram_counter, number_of_ngrams

def count_ngrams(texts, n):
	''' Takes a list of strings and count all n-grams in the entire list of text
		returns the total number of ngrams as integer, as well as a count for all unique ngrams as a cCounter object.
		Note that the number of n-grams in a text t is: len(t) -(n-1)
				'aabb', has len 4, contains 3 bigrams, 2 trigrams, 1 fourgram
	'''

	results = Counter()
	number_of_ngrams = 0
	for t in texts:
		ngram_counter, n_ng = count_ngram(t, n)
		number_of_ngrams += n_ng
		results += ngram_counter
	return results, number_of_ngrams



# get all ngram frequencies

texts = metadata.load_corpus_txt( train[did] )
ng_counter, number_of_ngrams = count_ngrams(texts, n)

frequencies = pd.DataFrame(ng_counter.items(), columns = ['ngram', 'freq'])
frequencies['freq'] = frequencies['freq'] / number_of_ngrams
frequencies = frequencies.sort('freq', ascending=False)

all_ngram = np.array(frequencies.ngram)
all_freq = np.array(frequencies.freq)



#Build ngram-frquencies for each document
print '### Build ngram frequencies for each document. n=', n

doc_id = list(test[did])
texts = metadata.load_corpus_txt( doc_id )
doc_freq = []
doc_ngram = []
for text in texts:
	ng_counter, number_of_ngrams = count_ngram(text, n)
	freq = pd.DataFrame(ng_counter.items(), columns = ['ngram', 'freq'])
	freq['freq'] = freq['freq'] / number_of_ngrams
	freq = freq.sort('freq', ascending=False)
	doc_freq.append(np.array(freq.freq))
	doc_ngram.append(np.array(freq.ngram))

# Build author profiles
# profiles and authors variables are interlinked. profiles[10] is the author profile (list of ngram frequencies) for authors[10], etc.

auth_freq = []
auth_ngram = []
for author in authors:
	a_texts = metadata.load_corpus_txt(train[train[aid]==author][did])
	ng_counter, number_of_ngrams = count_ngrams(a_texts, n)
	freq = pd.DataFrame(ng_counter.items(), columns = ['ngram', 'freq'])
	freq['freq'] = freq['freq'] / number_of_ngrams
	freq = freq.sort('freq', ascending=False)
	auth_freq.append(np.array(freq.freq))
	auth_ngram.append(np.array(freq.ngram))


#pickle.dump( variable_name, open('file.name', 'wb'))
#pickle.load( open('file.name', 'rb') )

pickle.dump( all_freq, open('ng_all_freq'+str(n)+'.p', 'wb'))
pickle.dump( doc_id, open('ng_doc_id'+str(n)+'.p', 'wb'))

pickle.dump( doc_ngram, open('ng_doc_ngram'+str(n)+'.p', 'wb'))
pickle.dump( doc_freq, open('ng_doc_freq'+str(n)+'.p', 'wb'))

pickle.dump( authors, open('ng_authors'+str(n)+'.p', 'wb'))
pickle.dump( auth_ngram, open('ng_auth_ngram'+str(n)+'.p', 'wb'))
pickle.dump( auth_freq, open('ng_auth_freq'+str(n)+'.p', 'wb'))