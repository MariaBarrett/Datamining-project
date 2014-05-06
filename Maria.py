from __future__ import division
import numpy as np
import glob
from bs4 import BeautifulSoup
import nltk
import lxml
import re
from collections import Counter
import math
import scipy

corpuspath = glob.glob('2539/CORPUS_UTF-8/*.xml')

"""
This function expects a path to the corpus files and extracts metadata, person features, lexical features
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

		#Lexical features
		temp.append(len(bodytext)) #Lex: total characters
		upper = sum(x.isupper() for x in bodytext)
		lower = sum(x.islower() for x in bodytext)
		temp.append(lower+upper) #Lex: total number of characters
		temp.append(upper) #Lex: upper case
		temp.append(sum(x.isdigit() for x in bodytext)) #Lex: number of digits
		temp.append(bodytext.count(' ')) #Lex: number of whitespace characters
		#can't find a way to count tabs
		lettercount = Counter(bodytext.lower())
		alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
		for letter in alphabet:
			temp.append(lettercount[letter] / len(bodytext))
		specialcharacters = ['~' , '@', '#', '$', '%', '^', '&', '*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '\\', '|', '(', ')']
		for char in specialcharacters:
			temp.append(lettercount[char] / len(bodytext))

		#Getting global variables
		tokenized = nltk.tokenize.word_tokenize(bodytext)
		alnum_tokenized = [word for word in tokenized if word.isalnum()]
		alnum_tokenized_lower = [word.lower() for word in alnum_tokenized] #only alphanumeric characters , lowercased
		textlenght = len(alnum_tokenized_lower)

		fdist = nltk.FreqDist(alnum_tokenized_lower)
		vocabulary = fdist.keys()
		numberofdifferentwords = len(vocabulary)

		#Word based features (WB)
		temp.append(textlenght) # Word-based: number of words
		temp.append(len([word for word in alnum_tokenized_lower if len(word)< 4])  / textlenght) # WB number of short words / number of words
		temp.append(sum([len(word) for word in alnum_tokenized_lower]) / len(bodytext)) # WB number of characters in words / number of characters
		temp.append(sum([len(word) for word in alnum_tokenized_lower]) / len (alnum_tokenized_lower)) #WB average word length

		sentences = bodyt.find_all('s')
		sum_char = 0
		sum_word = 0
		for sent in sentences:
			content = sent.contents[0]
			sum_char += len(content)
			sum_word += len(nltk.tokenize.word_tokenize(content))
		temp.append(sum_char / len(sentences)) #WB average characters per sentence
		temp.append(sum_word / len(sentences)) #WB average words per sentence

		temp.append(numberofdifferentwords) # WB number of different words
		temp.append(len([word for word in alnum_tokenized_lower if fdist[word] == 1]) / textlenght) # WB hapax legonema
		temp.append(len([word for word in alnum_tokenized_lower if fdist[word] == 2]) / textlenght) # WB hapax dislegonema
		
		# WB Lexical richness measures 
		
		#Yule's K #The larger the value the smaller the diversity
		inner = 0
		for i in xrange(1,21):
			inner += -(1/textlenght) + (len([word for word in alnum_tokenized_lower if fdist[word] == i]) * (i / textlenght)**2 )
		Yule = round(10**4 * inner)
		temp.append(Yule)

		#Simpson's D
		Simpson = 0
		for i in xrange(1,21):
			Simpson += ((len([word for word in alnum_tokenized_lower if fdist[word] == i]) * (i/textlenght) * ((i-1) / textlenght -1)))
		temp.append(Simpson)


		#Sichel's S - the higher the value the richer the text
		Sichel = len([word for word in alnum_tokenized_lower if fdist[word] == 2]) / textlenght #same as fraction of hapax dislegoma
		temp.append(Sichel)

		# WB Brunet's W
		W = len(alnum_tokenized)**(numberofdifferentwords**0.172) #getting W by use of a constant - a parametric method
		Brunet = (math.log(W) / math.log(len(alnum_tokenized)))**-0.172
		temp.append(Brunet)


		# Honores R / H ?? Uses hapax legomena
		Honore = round(100 * (math.log(textlenght) / (1-(len([word for word in alnum_tokenized_lower if fdist[word] == 1]) / numberofdifferentwords))))
		temp.append(Honore)

		#WB frequency of words of length 1 - 20
		for i in xrange(1,21):
			temp.append(len([word for word in alnum_tokenized_lower if len(word) == i])  / textlenght)

		#print temp
		
		paragraphs = bodyt.p
		paragraphtext = paragraphs.get_text() #use this

	dataset.append(temp)
	#np.asarray(dataset)

from_corpus(corpuspath)