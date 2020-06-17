# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:53:02 2020

@author: Andrzej T. Tunkiel

andrzej.t.tunkiel@uis.no

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = np.load('twd.npy',allow_pickle=True)

processed = []

for i in range(data.shape[0]):
    
    processed.append([])

    for j in range(data.shape[1]):
        
        processed[i].append([])

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range(len(data[i,j])):
           processed[i][j].append(np.mean(np.abs(data[i][j][k]))) 
           
#%%

for i, val_i in enumerate(processed):
    fig=plt.figure()
    for j, val_j in enumerate(val_i):
        plt.subplot(7,1,j+1)
        x = np.arange(0,len(val_j),1)
        plt.plot(x, val_j, label=j)
        plt.xlim(-5,100)
        plt.ylim(0,50)
        
    plt.legend()
    plt.show()
    
#%%


heat = []

for i in range(data.shape[0]):
    
    heat.append([])

    for j in range(data.shape[1]):
        
        heat[i].append([])

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        heat[i][j]=(np.mean(np.abs(np.hstack(data[i][j]))))
        
heat = np.asarray(heat)

av = np.average(heat,axis=1)
av = np.round(av,2)

manlist = [f'Gradient Boosting Regressor\nMAE={av[0]}',
 f'XGB Regressor\nMAE={av[1]}',
 f'Random Forest Regressor\nMAE={av[2]}',
 f'AdaBoost Regressor\nMAE={av[3]}',
f'KNeighbors Regressor\nMAE={av[4]}']

plt.figure(figsize=(6,3))
sns.heatmap(heat,annot=True, cmap='viridis',
            cbar_kws={'label': 'Mean Absolute Error [m/h]'})
plt.yticks(np.arange(0.5,len(heat),1),manlist, rotation=0)
plt.xlabel("Well evaluated")
plt.tight_layout()
plt.savefig('Cont_heatmap.pdf')

#%%

plt.plot(processed[0][3])