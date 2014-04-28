import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filepath ='2539/documentation/'
filename = 'BAWE.xls'

data = pd.read_excel(filepath+filename, 'Sheet1')

columns = ['words','s-units','p-units']

data.groupby(['disciplinary group'])[columns].describe()

for g in columns:
	data[g].hist(by=data['disciplinary group'])
	plt.suptitle(g)
	data.boxplot(column=g, by='disciplinary group')

plt.show()