import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random, glob


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

# sort files

def split_authors(data, threshold=3):
	''' strips authors with less than threshold articles. Split remaining article
	ID's into train and test sets, ensuring each author is represented
	with at least threshold-1 article in train and exactly 1 in test.
	Returns list of documents ID's (did) as lists '''


	student_id = data['student_id'].unique()

	train = []	# List of document ID's
	train_id = [] # List of student ID's
	test = []
	test_id = []

	for sid in student_id:
		dids = data[data['student_id']==sid]['id']
		length = len(dids)
		if length > threshold:
			dids = list(dids)
			r = random.randint(0,length-1)
			test.append(dids[r])
			test_id.append(sid)
			dids.pop(r)
			for d in dids:
				train.append(d)
				train_id.append(sid)
	return train, test, train_id, test_id

def load_corpus_txt(did):
	''' Takes a list of document id's (did) (for example from 
	split_authors()) and loads the content of the corrosponding txt
	files into a list and returns this. '''

	path = '2539/CORPUS_TXT/'
	fileext = '.txt'
	texts = []
	for d in did:
		texts.append(open(path+d+fileext).read().strip())
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






