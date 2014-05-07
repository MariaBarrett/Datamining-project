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
	metadata = []


	for xmlfile in path:
		#Assigning same id to metadata and dataset arrays
		i = 0
		#append document id to both lists 
		temp_dataset = []
		temp_metadata = []

		temp_dataset.append(i)
		temp_metadata.append(i)

		i+=1

		opened = open(xmlfile, 'r')	
		currentfile = BeautifulSoup(opened, 'xml')
		
		meta = currentfile.find('sourceDesc')
		metafeat = meta.find_all('p')
		for feat in metafeat:
			temp_metadata.append (feat.contents[0])

		persondata = currentfile.find('person')
		personfeat = persondata.find_all('p')
		for feat in personfeat:
			temp_metadata.append(feat.contents[0])

		bodyt = currentfile.find('body')
		bodytext = bodyt.get_text() #like the txt files

		#Lexical features
		allchars = len(bodytext)
		temp_dataset.append(allchars) #Lex: total characters
		upper = sum(x.isupper() for x in bodytext)
		lower = sum(x.islower() for x in bodytext)
		temp_dataset.append((lower+upper) / allchars) #Lex: total number of letters-characters
		temp_dataset.append(upper / allchars) #Lex: upper case
		temp_dataset.append(sum(x.isdigit() for x in bodytext) / allchars ) #Lex: number of digits
		temp_dataset.append(bodytext.count(' ') / allchars ) #Lex: number of whitespace characters
		#can't find a way to count tabs / text contains no tabs
		lettercount = Counter(bodytext.lower())
		alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
		for letter in alphabet:
			temp_dataset.append(lettercount[letter] / allchars)
		specialcharacters = ['~' , '@', '#', '$', '%', '^', '&', '*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '\\', '|', '(', ')']
		for char in specialcharacters:
			temp_dataset.append(lettercount[char] / allchars)

		#Getting global variables 
		tokenized = nltk.tokenize.word_tokenize(bodytext)
		alnum_tokenized = [word for word in tokenized if word.isalnum()]
		alnum_tokenized_lower = [word.lower() for word in alnum_tokenized] #only alphanumeric characters , lowercased
		num_allwords = len(alnum_tokenized_lower) #number of words

		fdist = nltk.FreqDist(alnum_tokenized_lower)
		vocabulary = fdist.keys() # all different words
		numberofdifferentwords = len(vocabulary)

		#Word based features (WB)
		temp_dataset.append(num_allwords) # Word-based: number of words
		temp_dataset.append(len([word for word in alnum_tokenized_lower if len(word)< 4])  / num_allwords) # WB number of short words / number of words
		temp_dataset.append(sum([len(word) for word in alnum_tokenized_lower]) / len(bodytext)) # WB number of characters in words / number of characters
		temp_dataset.append(sum([len(word) for word in alnum_tokenized_lower]) / len(alnum_tokenized_lower)) #WB average word length

		sentences = bodyt.find_all('s') #find all <s> tags
		sum_char = 0
		sum_word = 0
		# By counting words contained inside <s> tags we are not looking at headings or quotes. Length of headings/title may be interesting but content of quotes does not reveal author style
		for sent in sentences:
			content = sent.contents[0]
			sum_char += len(content)
			sum_word += len(nltk.tokenize.word_tokenize(content))
		temp_dataset.append(sum_char / len(sentences)) #WB average characters per sentence
		temp_dataset.append(sum_word / len(sentences)) #WB average words per sentence

		temp_dataset.append(numberofdifferentwords) # WB number of different words
		temp_dataset.append(len([word for word in alnum_tokenized_lower if fdist[word] == 1]) / num_allwords) # WB hapax legonema
		temp_dataset.append(len([word for word in alnum_tokenized_lower if fdist[word] == 2]) / num_allwords) # WB hapax dislegonema
		
		# WB Lexical richness measures


		#Yule's K #The larger the value the smaller the diversity. Larger value range
		inner = 0
		for i in xrange(1,21): #assumes max_word_len = 20
			inner += (len([word for word in alnum_tokenized_lower if fdist[word] == i]) * (i / num_allwords)**2 )
		inner -= (1/num_allwords)
		Yule = round(10**4 * inner)
		temp_dataset.append(Yule)

		#Simpson's D: The larger the value the larger the diversity. Smaller value range
		Simpson = 0
		for i in xrange(1,21):
			Simpson += ((len([word for word in alnum_tokenized_lower if fdist[word] == i]) * (i/num_allwords) * ((i-1) / num_allwords -1)))
		temp_dataset.append(Simpson)


		#Sichel's S - the higher the value the richer the text. Uses pretty small value range
		Sichel = len([word for word in alnum_tokenized_lower if fdist[word] == 2]) / numberofdifferentwords #same as fraction of hapax dislegoma
		temp_dataset.append(Sichel)

		# WB Brunet's W. The higher the value larger the diversity. Uses large value range
		Brunet = num_allwords**(numberofdifferentwords**0.172) #getting W by use of a constant - a parametric method
		temp_dataset.append(Brunet)

		# Honores R / H ?? Uses hapax legomena. The larger the value the smaller the diversity. Use pretty large value range
		Honore = 100 * (math.log(num_allwords) / (1-(len([word for word in alnum_tokenized_lower if fdist[word] == 1]) / numberofdifferentwords)))
		temp_dataset.append(Honore)

		#WB frequency of words of length 1 - 20
		for i in xrange(1,21):
			temp_dataset.append(len([word for word in alnum_tokenized_lower if len(word) == i])  / num_allwords)

		#print temp_dataset
		
		paragraphs = bodyt.p
		paragraphtext = paragraphs.get_text() #use this

	dataset.append(temp_dataset)
	return metadata, dataset
	#np.asarray(dataset)

from_corpus(corpuspath)