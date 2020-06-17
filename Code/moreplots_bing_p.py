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

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


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
av = np.zeros(100)

manlist = [f'Gradient Boosting Regressor\nMAPE={av[0]}',
 f'XGB Regressor\nMAPE={av[1]}',
 f'Random Forest Regressor\nMAPE={av[2]}',
 f'AdaBoost Regressor\nMAPE={av[3]}',
f'KNeighbors Regressor\nMAPE={av[4]}',
f'Bingham \nMAPE={av[5]}']







av_rop = np.load('ave_rop.npy')

ofa = np.asarray(np.load('ofa_singles.npy', allow_pickle=True))
ofa_bing = np.asarray(np.load('ofa_singles_bing.npy', allow_pickle=True))
ofa = np.concatenate((ofa, [ofa_bing]))


afo = np.asarray(np.load('afo_singles.npy', allow_pickle=True))
afo_bing = np.asarray(np.load('afo_singles_bing.npy', allow_pickle=True))
afo = np.concatenate((afo, [afo_bing]))



# afo_flat = np.zeros(afo.shape)
# for i in range(afo.shape[0]):
#     for j in range(afo.shape[1]):
#         afo_flat[i,j] = np.average(np.abs((afo[i,j])))
# afo_flat = afo_flat/av_rop*100
        
# ofa_flat = np.zeros(ofa.shape)
# for i in range(ofa.shape[0]):
#     for j in range(ofa.shape[1]):
#         ofa_flat[i,j] = np.average(np.abs((ofa[i,j])))
# ofa_flat = ofa_flat/av_rop*100




for i in range(len(afo)):
    for j in range(len(afo[i])):
        afo[i][j] = np.array(afo[i][j])/av_rop[j]*100

for i in range(len(manlist)):
    av[i] = np.round(np.mean(np.abs(np.hstack(afo[i]))),1)

afo_flat = np.zeros(afo.shape)
for i in range(afo.shape[0]):
    for j in range(afo.shape[1]):
        afo_flat[i,j] = np.average(np.abs((afo[i,j])))


manlist_afo = [f'Gradient Boosting Regressor\nMAPE={av[0]}',
 f'XGB Regressor\nMAPE={av[1]}',
 f'Random Forest Regressor\nMAPE={av[2]}',
 f'AdaBoost Regressor\nMAPE={av[3]}',
f'KNeighbors Regressor\nMAPE={av[4]}',
f'Bingham \nMAPE={av[5]}']

    



for i in range(len(ofa)):
    for j in range(len(ofa[i])):
        ofa[i][j] = np.array(ofa[i][j])/av_rop[j]*100

for i in range(len(manlist)):
    av[i] = np.round(np.mean(np.abs(np.hstack(ofa[i]))),1)

ofa_flat = np.zeros(ofa.shape)
for i in range(ofa.shape[0]):
    for j in range(ofa.shape[1]):
        ofa_flat[i,j] = np.average(np.abs((ofa[i,j])))

manlist_ofa = [f'Gradient Boosting Regressor\nMAPE={av[0]}',
 f'XGB Regressor\nMAPE={av[1]}',
 f'Random Forest Regressor\nMAPE={av[2]}',
 f'AdaBoost Regressor\nMAPE={av[3]}',
f'KNeighbors Regressor\nMAPE={av[4]}',
f'Bingham \nMAPE={av[5]}']
globalmax = np.max([afo_flat, ofa_flat])
globalmin = np.min([afo_flat, ofa_flat])

plt.figure(figsize=(5,3))
sns.heatmap(afo_flat, vmin=globalmin, vmax=globalmax, cmap='viridis',
            cbar_kws={'label': 'Mean Absolute Percentage Error [%]'}, annot=True,
            fmt=".1f")
plt.xlabel("Well left for testing")
plt.yticks(np.linspace(0.5,len(manlist_afo)-0.5,len(manlist_afo)), manlist_afo, rotation=0)
plt.tight_layout()
plt.savefig("AFO_heatmap_p.pdf")
plt.show()   

plt.figure(figsize=(5,3)) 
sns.heatmap(ofa_flat, vmin=globalmin, vmax=globalmax, cmap='viridis',
            cbar_kws={'label': 'Mean Absolute Percentage Error [%]'}, annot=True,
            fmt=".1f")
plt.xlabel("Well available for training")
plt.yticks(np.linspace(0.5,len(manlist_ofa)-0.5,len(manlist_ofa)), manlist_ofa, rotation=0)
plt.tight_layout()
plt.savefig("OFA_heatmap_p.pdf")
plt.show()

#%%
#afo_singles = np.load('afo_singles.npy', allow_pickle=True)
#ofa_singles = np.load('ofa_singles.npy', allow_pickle=True)



matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

afo_singles = afo
ofa_singles = ofa
av = np.zeros(len(manlist))
manlist = [f'Gradient\nBoosting\nRegressor\nMAE={av[0]}',
 f'XGB\nRegressor\n\nMAE={av[1]}',
 f'Random\nForest\nRegressor\nMAE={av[2]}',
 f'AdaBoost\nRegressor\n\nMAE={av[3]}',
f'KNeighbors\nRegressor\n\nMAE={av[4]}',
f'Bingham\n\n\nMAE={av[5]}']


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
av = np.zeros(len(manlist))

for i in range(len(manlist)):
    av[i] = np.round(np.mean(np.abs(np.hstack(afo_singles[i]))),2)



#sns.set_style(style="white",rc= {'patch.edgecolor': 'black'})
sns.violinplot(x=loc, y=np.abs(temp), color='silver')
#patch_violinplot()
plt.xticks(np.arange(0,len(manlist),1), manlist, rotation=0)
plt.ylim(-1,3)
plt.grid()
plt.ylabel("ROP MAE [m/h]")
plt.tight_layout()
plt.savefig('AFO_p.pdf')


plt.figure(figsize=(6,3))
    
loc = []
temp = []
for i in range(len(ofa_singles)):
    temp.append(np.hstack(ofa_singles[i]))
    
    loc.append(np.ones(len(np.hstack(ofa_singles[i])))*i)
    
loc = np.hstack(loc)
temp = np.hstack(temp)
av = np.zeros(len(manlist))

for i in range(len(manlist)):
    av[i] = np.round(np.mean(np.abs(np.hstack(ofa_singles[i]))),2)




sns.violinplot(x=loc, y=np.abs(temp), color="silver")
#patch_violinplot()
plt.xticks(np.arange(0,len(manlist),1), manlist, rotation=0)
plt.ylim(-1,3)
plt.ylabel("ROP MAE [m/h]")
plt.grid()
plt.tight_layout()
plt.savefig('OFA_p.pdf')

