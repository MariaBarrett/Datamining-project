from __future__ import division
import numpy as np
import glob
from bs4 import BeautifulSoup
import nltk
import lxml
import re
from nltk.corpus import stopwords
from collections import Counter

path = glob.glob('2539/CORPUS_UTF-8/*.xml')

"""
This function expects a path to the corpus files and extracts metadata and 
"""
tagmap = dict([tmap.strip("\n").split("\t") for tmap in open("en-ptb.map", 'r').readlines()])
funcwords = sorted([word.strip("\r\n") for word in open("en-func-words.txt", 'r').readlines() if len(nltk.word_tokenize(word)) == 1])
punctuations = [r""",""",r""".""",r"""?""",r"""!""",r""":""",r""";""",r"""'""",r""" " """[1]]
dataset = []

for xmlfile in path[0:1]:
	temp = []
	opened = open(xmlfile, 'r')	
	currentfile = BeautifulSoup(opened)
	
	bodyt = currentfile.find('body')
	bodytext = nltk.word_tokenize(bodyt.get_text())
	ptb_tag_text = nltk.pos_tag(bodytext) # Tagging the text with PTB tags
	uni_tag_text = [(wt[0],tagmap[wt[1]]) for wt in ptb_tag_text if wt[1] in tagmap.keys()] # Mapping: PTB to uni tags
	freq_dist = nltk.FreqDist(wt[1] for wt in uni_tag_text) # NLTK frequency distribution over uni tags
	tag_freq = [freq_dist.freq(uni_tag) for uni_tag in sorted(list(set(tagmap.values())))] # Frequency of uni tags
	func_freq = [bodytext.count(fw) for fw in funcwords] # Count the function words in the text
	punc_freq = [bodytext.count(p)/len(bodyt.get_text()) for p in punctuations] # Count the punctuations in the text relative to overall number of characters







