# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:07:57 2020

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no

"""

# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob


from scipy.optimize import least_squares
import seaborn as sns

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
#%%
#Get list of files from directory
filelist = glob(r'USROP*.csv', recursive=False)

print ("Detected logs:\n")

for i in range(len(filelist)):
    print ('[' + str(i) + ']' + " " +
           filelist[i].split('\\')[-1] +
           " " + str(os.path.getsize(filelist[i])//1000000) + 'MB')

print ()

dfa = []
filelist.sort()
for i in filelist:
    dfa.append(pd.read_csv(i))
#%%

# for i, val in enumerate(dfa):
#     sns.distplot(val['Rate of Penetration m/h'], bins=10)
#     plt.xlim(0,100)
#     plt.show()
    
    
#%%
df = pd.DataFrame()
for i, val in enumerate(dfa):
    val['Well'] = str(i) + "\nMean = " + str(np.round(np.average(val['Rate of Penetration m/h']),))
    df = df.append(val)
    
for i, val in enumerate(dfa):
    val['Well'] = 'All' 
    df = df.append(val)
       
avall = np.round(np.average(df[df['Well'] == 'All']['Rate of Penetration m/h']),1)

df = df.replace('All', "All\nMean = " + str(avall))
#%%
#sns.set_style("whitegrid")
bins = np.arange(0,101,5)
g = sns.FacetGrid(df, col="Well", height=1.7, aspect=1, col_wrap=4)
g = g.map(plt.hist, 'Rate of Penetration m/h', bins=bins, log=True,
          edgecolor='black', linewidth=1,color="grey")

axes = g.axes
for i in axes:
    i.set_xlim(0,100)
    i.set_ylim(10,30000)
    i.set_yscale('log')
    i.set_xticks(np.arange(0,101,25))
    i.set_yticks([10,100,1000,10000])
    i.set_xticklabels(labels=np.arange(0,101,25), rotation=90)
    i.grid(color="lightgrey", linestyle='-', linewidth=1)

plt.tight_layout()
plt.savefig("ROP_distribution.pdf")
#%%
