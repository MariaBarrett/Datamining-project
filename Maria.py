from __future__ import division
import numpy as np
import glob
from bs4 import BeautifulSoup
import nltk 

corpuspath = glob.glob('2539/CORPUS_UTF-8/*.xml')

def from_corpus(path):
	for xmlfile in path:
		opened = open(xmlfile, 'r')
		currentfile = BeautifulSoup(opened, 'xml')
		metadata = currentfile.find('sourceDesc')
		metadatafeat = metadata.find_all('p')
		persondata = currentfile.find('person')
		personfeat = persondata.find_all('p')
		bodyt = currentfile.find('body')
		bodytext = bodyt.get_text()

		print personfeat
from_corpus(corpuspath)