import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from collections import Counter
import random, pickle


filepath ='2539/documentation/'
filename = 'BAWE.xls'

data = pd.read_excel(filepath+filename, 'Sheet1')


def build_hist_box(data):
	columns = ['words','s-units','p-units']

	data.groupby(['disciplinary group'])[columns].describe()

	for g in columns:
		data[g].hist(by=data['disciplinary group'])
		plt.suptitle(g)
		data.boxplot(column=g, by='disciplinary group')

	plt.show()

def split(data, split=0.7):
	''' splits documents into a train and test set. returns 2 lists of document ids.
	'''
	doc_id = np.array(data['id'])
	l = len(doc_id)
	index = range(l)
	np.random.shuffle(index)
	split = int(l*split)
	
	train = doc_id[index[:split]]
	test = doc_id[index[split:]]

	return train, test

def split_authors_pd(data, threshold=4, minimum=2, max_authors=100):
	''' strips authors with less than 3 (or threshold) articles. Split remaining article
	ID's into train and test sets, ensuring each author is represented
	with at least 1 article in train and exactly minimum in test.
	Returns pandas dataframe containing all author ids('aid') and document ids('did'''

	student_id = data['student_id'].unique()

	no_of_doc = []
	for sid in student_id:
		no_of_doc.append(len(data[data.student_id == sid]))

	index = np.argsort(no_of_doc)

	student_id = student_id[index[-max_authors:]]

	# Sanity check. Code won't work with threshold less than 2!
	if threshold < minimum:
		threshold = 2

	columns = ['student_id', 'id']
	test = pd.DataFrame(columns=columns)
	train = test[:]

	author_ids = student_id

	for aid in author_ids:
		rows = data[data['student_id']== aid][['student_id','id']]
		length = len(rows)

		if length >= threshold:
			rng = range(length)
			for i in range(minimum):
				r = random.randint(0,len(rng)-1)
				test = test.append(rows.iloc[r]) #test is a pandas DataFrame, so append doesn't mutate the object!
				rng.pop(r)
			train = train.append(rows.iloc[rng])
	return train, test

def stripAllTags( html ):
	if html is None:
		return None
	return ''.join( BeautifulSoup( html ).findAll( text = True ) ) 

def load_corpus_txt(did):
	''' Takes a list of document id's (did) (for example from 
	split_authors()) and loads the content of the corrosponding txt
	files into a list and returns this. '''

	path = '2539/CORPUS_TXT/'
	fileext = '.txt'
	texts = []
	for d in did:
		texts.append(stripAllTags(open(path+d+fileext).read().strip()))
	return texts

def get_all_unique_chars(texts, verbose=False):
	''' Takes a list of texts and return all unique
	characters as a list. This list is useful for building
	character level n-grams '''

	alltext = ''
	lengths = []
	for t in texts:
		alltext = alltext + t
		lengths.append(len(set(t)))

	unique = list(set(alltext))

	if verbose == True:
		print 'max length:', max(lengths)
		print 'min length:', min(lengths)
		print 'average length', mean(lengths)
		print 'unique characters', len(unique)

	return unique