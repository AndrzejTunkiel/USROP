# -*- coding: utf-8 -*-
"""
Created on Wed May 27 12:51:09 2020

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
#Evaluating various standard regressors
    

from sklearn.metrics import mean_absolute_error as MAE


def bingham(K, a, WOB, D, N):
    return N*K*(WOB/D)**a


def bingham_opt(par, WOB, D, N, ROP):
    K = par[0]
    a = par[1]
    return N*K*(WOB/D)**a - ROP

results_singles = [[],[],[],[],[],[],[]]
results_singles_p = [[],[],[],[],[],[],[]]

for i in range(len(dfa)):


    increment = 577
    depth = increment
    
    while True:
        

        #print(f'''Well: {i},  depth: {
        #    np.round(depth/dfa[i].index.max()*100,1)}%''')
        train = dfa[i][0:depth]
        
        WOB = train['Weight on Bit kkgf'].to_numpy()
        D = train['Diameter mm'].to_numpy()
        N = train['Average Rotary Speed rpm'].to_numpy()
        ROP = train['Rate of Penetration m/h'].to_numpy()

        params = np.ones(2) #initiating the parameters with ones

        model = least_squares(bingham_opt, params,
                              args=(WOB, D, N, ROP)
                              )

        test = dfa[i][depth:depth+increment]
        
        WOB = test['Weight on Bit kkgf'].to_numpy()
        D = test['Diameter mm'].to_numpy()
        N = test['Average Rotary Speed rpm'].to_numpy()
        ROP = test['Rate of Penetration m/h'].to_numpy()
        K = model.x[0]
        a = model.x[1]
        
        local_result = []
        local_result_p = []
        
        for j in range(len(ROP)):
            local_result.append(ROP[j] - bingham(K, a, WOB[j], D[j], N[j]))
            local_result_p.append((ROP[j] - bingham(K, a, WOB[j], D[j], N[j]))/ROP[j])
        
        results_singles[i].append(local_result)
        results_singles_p[i].append(local_result_p)
        depth = depth + increment
        if depth >= dfa[i].index.max():
            break
        
np.save("twd_bing.npy", results_singles)   
np.save("twd_p_bing.npy", results_singles_p)            
#%%
