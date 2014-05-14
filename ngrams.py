'''
Implement Keselj, Vlado, et al.'s n-gram author attribution on BAWE
'''

import metadata
import numpy as np
import matplotlib.pyplot as plt

data = metadata.data
profile_length = 159**2

train_did, test_did, test_id, test_id = metadata.split_authors(data)

train = metadata.load_corpus_txt(train_index)
test = metadata.load_corpus_txt(test_index)

authors = data['student_id'].unique()

profiles = []