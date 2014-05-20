from __future__ import division
from collections import Counter
from featuremap import featuremap, metadatamap
import matplotlib.pyplot as plt
from sklearn import svm, tree, neighbors, grid_search
import numpy as np
from scipy import stats
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
import func, pickle
'''
Generates a series of plots and summary statistics of the entire dataset.
Generates a series of plots of the best features for grade, level and native language tasks based on the training set.
'''

# Load data

metadata = pickle.load( open( "metadata.p", "rb" ) )
data = pickle.load( open( "dataset.p", "rb" ) )
plusone = np.where(metadata[:,9] != "1")[0] # Get indexes for all texts written by more than one person
data, metadata = np.delete(data, plusone, 0), np.delete(metadata, plusone, 0) # Remove all texts written by more than one person from data

# RAW DATA SETS
NLtrain_y, NLtrain_X, NLtest_y, NLtest_X = func.natlan(metadata,data)
GRtrain_y, GRtrain_X, GRtest_y, GRtest_X = func.grade(metadata,data)
LEtrain_y, LEtrain_X, LEtest_y, LEtest_X = func.level(metadata,data)
AUtrain_y20, AUtrain_X20, AUtest_y20, AUtest_X20 = func.author(metadata,data,4,20,2)
AUtrain_yAH, AUtrain_XAH, AUtest_yAH, AUtest_XAH = func.author(metadata,data,4,20,2,(7,"AH"))
AUtrain_y100, AUtrain_X100, AUtest_y100, AUtest_X100 = func.author(metadata,data,4,100,2)

# NORMALIZED DATA SETS
NLtrain_Xn, NLtest_Xn = func.normalize(NLtrain_X, NLtest_X)
GRtrain_Xn, GRtest_Xn = func.normalize(GRtrain_X, GRtest_X)
LEtrain_Xn, LEtest_Xn = func.normalize(LEtrain_X, LEtest_X)
AUtrain_Xn20, AUtest_Xn20 = func.normalize(AUtrain_X20, AUtest_X20)
AUtrain_XnAH, AUtest_XnAH = func.normalize(AUtrain_XAH, AUtest_XAH)
AUtrain_Xn100, AUtest_Xn100 = func.normalize(AUtrain_X100, AUtest_X100)


filepath ='2539/documentation/'
filename = 'BAWE.xls'

mdata = pd.read_excel(filepath+filename, 'Sheet1')
# Clean data
mdata.grade[mdata.grade==' ']= 'unknown'
mdata.L1[mdata.L1 != 'English'] = 'Other'

#----------------------------------------------------------------------------------------
# Begin descriptive
#    FUNCTIONS
#----------------------------------------------------------------------------------------

def build_hist_box(data, groupby='disciplinary group', columns = ['words','s-units','p-units']):

	data.groupby([groupby])[columns].describe()

	for g in columns:
		data[g].hist(by=data[groupby])
		plt.suptitle(g+' grouped by '+groupby)
		data.boxplot(column=g, by=groupby)
	plt.show()


#----------------------------------------------------------------------------------------
# Begin descriptive
#----------------------------------------------------------------------------------------

#build_hist_box(mdata, groupby='level')
#build_hist_box(mdata, groupby='grade')

print mdata[['words', 's-units', 'p-units']].describe()
print mdata[['words', 's-units', 'p-units', 'L1']].groupby('L1').describe()
print mdata[['words', 's-units', 'p-units', 'grade']].groupby('grade').describe()
print mdata[['words', 's-units', 'p-units', 'level']].groupby('level').describe()

# top 10 features for native language
plt.figure()
NL = ['SYN_frac_while', 'SYN_frac_this', 'SYN_frac_besides', 'LEX_frac_h', 'SYN_frac_since', 'SYN_frac_nevertheless', 'SYN_frac_among', 'SYN_frac_might', 'SYN_frac_whilst', 'SYN_frac_a']
NL_index = [featuremap.index(i) for i in NL]

# get all columns in NL from data where 'first language' == 'English'
english = NLtrain_Xn[NLtrain_y == 1][:,NL_index]
# get all columns in NL from data where 'first language' == 'Other'
other = NLtrain_Xn[NLtrain_y == 0][:,NL_index]

everything = []
new_nl = []
nl_mean = []
for i, t in enumerate(NL):
	everything.append(english[:,i])
	everything.append(other[:,i])
	nl_mean.append(english[:,i].mean())
	nl_mean.append(other[:,i].mean())
	new_nl.append(t+'ENG')
	new_nl.append(t+'OTH')
plt.plot([0]*22, 'y')
plt.ylim([-4.5,17])
plt.boxplot(everything)
plt.plot(range(1,21),nl_mean, 'o')
plt.xticks(range(1,21), new_nl, size='small', rotation=45, horizontalalignment='right')
plt.title('top 10 features for Native Language (eng vs oth)')


# top 10 features for grade
plt.figure()
GR = ['SYN_frac_their', 'LEX_frac_-', 'SYN_frac_POS_PRON', 'LEX_frac_(', 'LEX_frac_)', 'SYN_frac_all_funcwords', 'SYN_frac_they', 'SYN_frac_that', 'SYN_frac_POS_PRT', 'LEX_frac_word_len8']

GR_index = [featuremap.index(i) for i in GR]

# 1 = M, 0 = D
# get all columns in NL from data where 'first language' == 'English'
M = GRtrain_Xn[GRtrain_y == 1][:,GR_index]
# get all columns in GR from data where 'first language' == 'Other'
D = GRtrain_Xn[GRtrain_y == 0][:,GR_index]

everything = []
new_nl = []
nl_mean = []
for i, t in enumerate(GR):
	everything.append(M[:,i])
	everything.append(D[:,i])
	nl_mean.append(M[:,i].mean())
	nl_mean.append(D[:,i].mean())
	new_nl.append(t+' M')
	new_nl.append(t+' D')
plt.plot([0]*22, 'y')
plt.ylim([-4.5,10])
plt.boxplot(everything)
plt.plot(range(1,21),nl_mean, 'o')
plt.xticks(range(1,21), new_nl, size='small', rotation=45, horizontalalignment='right')
plt.title('top 10 features for grade (M vs D)')




# top 5 features for Academic level

plt.figure()
LE = [
'LEX_num_characters', 
'LEX_num_words', 
'LEX_num_dif_words', 
'LEX_frac_whitesp', 
'SYN_frac_all_funcwords', 
]

LE_index = [featuremap.index(i) for i in LE]
LE_classes = {}

for i in range(1,5):
	# get all columns in LE from data where level == 1,2,3 or 4
	LE_classes[i] = LEtrain_Xn[LEtrain_y == i][:,LE_index]

everything = []
new_LE = []
LE_mean = []
for i, t in enumerate(LE):
	for j in LE_classes:
		everything.append(LE_classes[j][:,i])
		LE_mean.append(LE_classes[j][:,i].mean())
		new_LE.append(t+' lvl'+str(j))
plt.plot([0]*22, 'y')
plt.ylim([-4.5,8])
plt.boxplot(everything)
plt.plot(range(1,21),LE_mean, 'o')
plt.xticks(range(1,21), new_LE, size='small', rotation=45, horizontalalignment='right')
plt.title('top 5 features for Academic level')



