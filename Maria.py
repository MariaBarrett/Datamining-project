from __future__ import division
import numpy as np
import glob
from bs4 import BeautifulSoup
import nltk 
import lxml
import re
from collections import Counter

corpuspath = glob.glob('2539/CORPUS_UTF-8/*.xml')

"""
This function expects a path to the corpus files and extracts metadata and 
"""
def from_corpus(path):
	dataset = []
	for xmlfile in path:
		temp = []
		opened = open(xmlfile, 'r')	
		currentfile = BeautifulSoup(opened, 'xml')
		
		meta = currentfile.find('sourceDesc')
		metafeat = meta.find_all('p')
		for feat in metafeat:
			temp.append (feat.contents[0])

		persondata = currentfile.find('person')
		personfeat = persondata.find_all('p')
		for feat in personfeat:
			temp.append(feat.contents[0])

		bodyt = currentfile.find('body')
		bodytext = bodyt.get_text() #like the txt files
		temp.append(len(bodytext)) #Lex total characters
		upper = sum(x.isupper() for x in bodytext)
		lower = sum(x.islower() for x in bodytext)
		temp.append(lower+upper) #Lex total number of characters
		temp.append(upper) #Lex upper case
		temp.append(sum(x.isdigit() for x in bodytext)) #Lex number of digits
		temp.append(bodytext.count(' ')) #Lex number of whitespace characters
		#can't find a way to count tabs
		lettercount = Counter(bodytext.lower())
		alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
		for letter in alphabet:
			temp.append(lettercount[letter] / len(bodytext))
		specialcharacters = ['~' , '@', '\#', '\$', '\%', '\^', '\&', '\*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '\\', '\|', '(', ')']
		for char in specialcharacters:
			temp.append(lettercount[char] / len(bodytext))

		print temp
		paragraphs = bodyt.p
		paragraphtext = paragraphs.get_text() #use this

		sentences = bodyt.find('s')
		sentencetext = sentences.get_text() #use this
	dataset.append(temp)
	np.asarray(dataset)

from_corpus(corpuspath)