# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:16:23 2020

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no

"""
#%%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.svm import SVR

import xgboost as xgb



regs = {
        "GradientBoostingRegressor" : GradientBoostingRegressor(),
        "XGBRegressor" : xgb.XGBRegressor(),
        "RandomForestRegressor" : RandomForestRegressor(),
        "AdaBoostRegressor" : AdaBoostRegressor(),
        "KNeighborsRegressor" : KNeighborsRegressor(),

        }
#%%


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

manlist = [f'Gradient\nBoosting\nRegressor',
 f'XGB\nRegressor',
 f'Random\nForest\nRegressor',
 f'AdaBoost\nRegressor',
f'KNeighbors\nRegressor']


ofa = np.asarray(np.load('ofa_singles.npy', allow_pickle=True))
afo = np.asarray(np.load('afo_singles.npy', allow_pickle=True))

afo_flat = np.zeros(afo.shape)
for i in range(afo.shape[0]):
    for j in range(afo.shape[1]):
        afo_flat[i,j] = np.average(np.abs((afo[i,j])))
        
ofa_flat = np.zeros(ofa.shape)
for i in range(ofa.shape[0]):
    for j in range(ofa.shape[1]):
        ofa_flat[i,j] = np.average(np.abs((ofa[i,j])))

globalmax = np.max([afo_flat, ofa_flat])
globalmin = np.min([afo_flat, ofa_flat])

plt.figure(figsize=(6,3.5))
sns.heatmap(afo_flat, vmin=globalmin, vmax=globalmax, cmap='viridis',
            cbar_kws={'label': 'Mean Absolute Error [m/h]'}, annot=True)
plt.xlabel("Well left for testing")
plt.yticks(np.linspace(0.5,4.5,5), manlist, rotation=0)
plt.tight_layout()
plt.savefig("AFO_heatmap.pdf")
plt.show()       

plt.figure(figsize=(6,3.5)) 
sns.heatmap(ofa_flat, vmin=globalmin, vmax=globalmax, cmap='viridis',
            cbar_kws={'label': 'Mean Absolute Error [m/h]'}, annot=True)
plt.xlabel("Well available for training")
plt.yticks(np.linspace(0.5,4.5,5), manlist, rotation=0)
plt.tight_layout()
plt.savefig("OFA_heatmap.pdf")
plt.show()

#%%
afo_singles = np.load('afo_singles.npy', allow_pickle=True)
ofa_singles = np.load('ofa_singles.npy', allow_pickle=True)

def patch_violinplot():
     from matplotlib.collections import PolyCollection
     ax = plt.gca()
     for art in ax.get_children():
          if isinstance(art, PolyCollection):
              art.set_edgecolor((0.1, 0.1, 0.1))

import seaborn as sns

plt.figure(figsize=(6,3))
    
loc = []
temp = []
for i in range(len(afo_singles)):
    temp.append(np.hstack(afo_singles[i]))
    
    loc.append(np.ones(len(np.hstack(afo_singles[i])))*i)
    
loc = np.hstack(loc)
temp = np.hstack(temp)
av = np.zeros(len(regs))

for i in range(len(regs)):
    av[i] = np.round(np.mean(np.abs(np.hstack(afo_singles[i]))),2)

manlist = [f'Gradient\nBoosting\nRegressor\nMAE={av[0]}',
 f'XGB\nRegressor\n\nMAE={av[1]}',
 f'Random\nForest\nRegressor\nMAE={av[2]}',
 f'AdaBoost\nRegressor\n\nMAE={av[3]}',
f'KNeighbors\nRegressor\n\nMAE={av[4]}']
#alternatively use list(regs.keys())

sns.set_style(style="white",rc= {'patch.edgecolor': 'black'})
sns.violinplot(x=loc, y=np.abs(temp), color='silver')
#patch_violinplot()
plt.xticks(np.arange(0,len(regs),1), manlist, rotation=0)
plt.ylim(-5,60)
plt.grid()
plt.ylabel("ROP MEA [m/h]")
plt.tight_layout()
plt.savefig('AFO.pdf')


plt.figure(figsize=(6,3))
    
loc = []
temp = []
for i in range(len(ofa_singles)):
    temp.append(np.hstack(ofa_singles[i]))
    
    loc.append(np.ones(len(np.hstack(ofa_singles[i])))*i)
    
loc = np.hstack(loc)
temp = np.hstack(temp)
av = np.zeros(len(regs))

for i in range(len(regs)):
    av[i] = np.round(np.mean(np.abs(np.hstack(ofa_singles[i]))),2)

manlist = [f'Gradient\nBoosting\nRegressor\nMAE={av[0]}',
 f'XGB\nRegressor\n\nMAE={av[1]}',
 f'Random\nForest\nRegressor\nMAE={av[2]}',
 f'AdaBoost\nRegressor\n\nMAE={av[3]}',
f'KNeighbors\nRegressor\n\nMAE={av[4]}']
#alternatively use list(regs.keys())


sns.violinplot(x=loc, y=np.abs(temp), color="silver")
#patch_violinplot()
plt.xticks(np.arange(0,len(regs),1), manlist, rotation=0)
plt.ylim(-5,60)
plt.ylabel("ROP MEA [m/h]")
plt.grid()
plt.tight_layout()
plt.savefig('OFA.pdf')