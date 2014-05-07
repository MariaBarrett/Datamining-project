import nltk
import lxml
import re
from collections import Counter
import math

notdiverse = "this is a text this is a text this is a text this is a text this is a text this is a text this is a text. It iss not diverse"
diverse = "this is a diverse text. Many words are different. that means we get a high or low score in vocabulary richness. aa ab abc abcd abcde aa ab abc"

def global_var(rawtext):
	tokenized = nltk.tokenize.word_tokenize(rawtext)
	alnum_tokenized = [word for word in tokenized if word.isalnum()]
	alnum_tokenized_lower = [word.lower() for word in alnum_tokenized] #only alphanumeric characters , lowercased
	num_allwords = len(alnum_tokenized_lower) #number of words

	fdist = nltk.FreqDist(alnum_tokenized_lower)
	vocabulary = fdist.keys() # all different words
	numberofdifferentwords = len(vocabulary)
	return alnum_tokenized_lower, fdist, num_allwords, numberofdifferentwords


tokenized1, freqd1, txtlen1, difwords1 = global_var(notdiverse)
tokenized2, freqd2, txtlen2, difwords2 = global_var(diverse)


inner = 0
for i in xrange(1,21): #assumes max_word_len = 20
	inner += (len([word for word in tokenized1 if freqd1[word] == i]) * (i / txtlen1)**2 )
inner -= (1/txtlen1)
Yule1 = round(10**4 * inner)
print "Yule1", Yule1

inner = 0
for i in xrange(1,21): #assumes max_word_len = 20
	inner += (len([word for word in tokenized2 if freqd2[word] == i]) * (i / txtlen2)**2 )
inner -= (1/txtlen2)
Yule2 = round(10**4 * inner)
print "Yule2", Yule2


Simpson1 = 0
for i in xrange(1,21):
	Simpson1 += ((len([word for word in tokenized1 if freqd1[word] == i]) * (i/txtlen1) * ((i-1) / txtlen1 -1)))
print "Simpson1", Simpson1


Simpson2 = 0
for i in xrange(1,21):
	Simpson2 += ((len([word for word in tokenized2 if freqd2[word] == i]) * (i/txtlen2) * ((i-1) / txtlen2 -1)))
print "Simpson2", Simpson2


Sichel = len([word for word in tokenized1 if freqd1[word] == 2]) / txtlen1 #same as fraction of hapax dislegoma
print "Sichel1", Sichel

Sichel = len([word for word in tokenized2 if freqd2[word] == 2]) / txtlen2 #same as fraction of hapax dislegoma
print "Sichel2", Sichel

W = len(tokenized1)**(difwords1**0.172) #getting W by use of a constant - a parametric method
Brunet = (math.log(W) / math.log(len(tokenized1)))**-0.172
print "Brunet1", Brunet

W = len(tokenized1)**(difwords2**0.172) #getting W by use of a constant - a parametric method
print "Brunet2", W

Honore = round(100 * (math.log(len(tokenized1)) / (1-(len([word for word in tokenized1 if freqd1[word] == 1]) / difwords1))))
print "Honore1", Honore

Honore = round(100 * (math.log(len(tokenized2)) / (1-(len([word for word in tokenized2 if freqd2[word] == 1]) / difwords2))))
print "Honore2", W