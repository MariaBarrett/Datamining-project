import numpy as np
import pandas as pd

filepath ='2539/2539/documentation/'
filename = 'BAWE.xls'

data = pd.read_excel(filepath+filename, 'Sheet1')

dg = data['disciplinary group']
groups = list(set(dg))
groups.pop(3) # pops the nan value

for g in groups:
    print '###', g, '###'
    print data[data['disciplinary group']==g][['words','s-units','p-units']].describe()