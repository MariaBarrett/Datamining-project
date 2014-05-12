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
import pickle
import featuremap #our own feature map of all the features

corpuspath = glob.glob('2539/CORPUS_UTF-8/*.xml')

"""
This function expects a path to the corpus files and extracts metadata, person features, lexical features
"""
def from_corpus(path):
	dataset = []
	metadata = []

	tagmap = dict([tmap.strip("\n").split("\t") for tmap in open("en-ptb.map", 'r').readlines()])
	funcwords = sorted(list(set([word.strip("\r\n") for word in open("en-func-words.txt", 'r').readlines() if len(nltk.word_tokenize(word)) == 1])))
	punctuations = [',','.','?','!',':',';','\'','\"']
	specialcharacters = ['~' , '@', '#', '$', '%', '^', '&', '*', '-', '_', '=' ,'+', '>', '<', '[', ']', '{', '}', '/', '\\', '|', '(', ')']
	alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

	for xmlfile in path[:2]:
		
		temp_dataset = []
		temp_metadata = []

		opened = open(xmlfile, 'r')	
		currentfile = BeautifulSoup(opened, 'xml')
		
		#Document ID is first value in meta list
		temp_metadata.append(currentfile.find(True, {'id': True})['id'][1:])

		metafeat = currentfile.find('sourceDesc').find_all('p')
		for feat in metafeat:
			temp_metadata.append (feat.contents[0])

		personfeat = currentfile.find('person').find_all('p')
		for feat in personfeat:
			temp_metadata.append(feat.contents[0])

		bodyt = currentfile.find('body')
		bodytext = bodyt.get_text() #like the txt files

		#Lexical features
		allchars = len(bodytext)
		temp_dataset.append(float(allchars)) #Lex: total characters
		upper = sum(x.isupper() for x in bodytext)
		lower = sum(x.islower() for x in bodytext)
		temp_dataset.append((lower+upper) / allchars) #Lex: total number of letters-characters
		temp_dataset.append(upper / allchars) #Lex: upper case
		temp_dataset.append(sum(x.isdigit() for x in bodytext) / allchars ) #Lex: number of digits
		temp_dataset.append(bodytext.count(' ') / allchars ) #Lex: number of whitespace characters
		#can't find a way to count tabs / text contains no tabs
		lettercount = Counter(bodytext.lower())
		
		for letter in alphabet:
			temp_dataset.append(lettercount[letter] / allchars)
		
		for char in specialcharacters:
			temp_dataset.append(lettercount[char] / allchars)

		#Getting global variables 
		tokenized = nltk.tokenize.word_tokenize(bodytext)
		alnum_tokenized_lower = [word.lower() for word in tokenized if word.isalnum()] #only alphanumeric characters , lowercased
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

		
		# Syntactic features
		ptb_tag_text = nltk.pos_tag(tokenized) # Tagging the text with PTB tags
		uni_tag_text = [(wt[0],tagmap[wt[1]]) for wt in ptb_tag_text if wt[1] in tagmap.keys()] # Mapping: PTB to uni tags
		freq_dist = nltk.FreqDist(wt[1] for wt in uni_tag_text) # NLTK frequency distribution over uni tags
		tag_freq = [freq_dist.freq(uni_tag) for uni_tag in sorted(list(set(tagmap.values())))] # Frequency of uni tags
		func_freq = [alnum_tokenized_lower.count(fw)/num_allwords for fw in funcwords] # Count the function words in the text
		punc_freq = [bodytext.count(p)/allchars for p in punctuations] # Count the punctuations in the text relative to overall number of characters

		temp_dataset += punc_freq+func_freq+[sum(func_freq)]+tag_freq

		# Structural features
		sents_no = len(bodyt.find_all('s'))
		temp_dataset.append(sents_no) #number of sentences
		paragraphs_no = len(bodyt.find_all('p'))
		temp_dataset.append(paragraphs_no) #number of paragraphs

		temp_dataset.append(sents_no / paragraphs_no) # number of sentences per paragraph
		temp_dataset.append(allchars / paragraphs_no) # number of characters per paragraph
		temp_dataset.append(num_allwords / paragraphs_no) # number of words per paragraph
		blockquotes = len(bodyt.find_all('quote')) #number of block quotes

		#Appending to array
		dataset.append(temp_dataset)
		metadata.append(temp_metadata)


	return np.array(metadata), np.array(dataset)

metadata, dataset = from_corpus(corpuspath)
pickle.dump(metadata, open( "metadata.p", "wb" ) )
pickle.dump(dataset, open( "dataset.p", "wb" ) )

