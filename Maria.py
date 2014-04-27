from __future__ import division
import numpy as np
import glob
from bs4 import BeautifulSoup

corpuspath = glob.glob("2539/CORPUS_UTF-8/*.xml")

def from_corpus(path):
	for xmlfile in path:
		currentfile = BeautifulSoup(xmlfile)
		metadata = currentfile.find("sourceDesc")