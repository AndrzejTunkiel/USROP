# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob
from scipy.optimize import least_squares

#%%
#Get list of files from directory
filelist = glob(r'USROP_A*.csv', recursive=False)

print ("Detected logs:\n")

for i in range(len(filelist)):
    print ('[' + str(i) + ']' + " " +
           filelist[i].split('\\')[-1] +
           " " + str(os.path.getsize(filelist[i])//1000000) + 'MB')

print ()

dfa = []

for i in filelist:
    dfa.append(pd.read_csv(i))
    

#%%
#Evaluating various standard regressors

def bingham(K, a, WOB, D, N):
    return N*K*(WOB/D)**a

def bingham_opt(par, WOB, D, N, ROP):
    K = par[0]
    a = par[1]
    return N*K*(WOB/D)**a - ROP
#%%


afo_singles = []
afo_singles_percentage = []

for j in range(len(dfa)):
    dfa_train = dfa.copy()
    test = dfa_train.pop(j)
    
    train = pd.concat(dfa_train)
    
    WOB = train['Weight on Bit kkgf'].to_numpy()
    D = train['Diameter mm'].to_numpy()
    N = train['Average Rotary Speed rpm'].to_numpy()
    ROP = train['Rate of Penetration m/h'].to_numpy()

    params = np.ones(2) #initiating the parameters with ones

    model = least_squares(bingham_opt, params,
                          args=(WOB, D, N, ROP)
                          )

    WOB = test['Weight on Bit kkgf'].to_numpy()
    D = test['Diameter mm'].to_numpy()
    N = test['Average Rotary Speed rpm'].to_numpy()
    ROP = test['Rate of Penetration m/h'].to_numpy()
    K = model.x[0]
    a = model.x[1]
    
    local_result = []
    local_result_percentage = []
    
    for j in range(len(ROP)):
        local_result.append(ROP[j] - bingham(K, a, WOB[j], D[j], N[j]))
        
    for j in range(len(ROP)):
        local_result_percentage.append((ROP[j] - bingham(K, a, WOB[j], D[j], N[j]))/ROP[j])

    afo_singles.append(local_result)
    afo_singles_percentage.append(local_result_percentage)
        
        

np.save("afo_singles_bing.npy", afo_singles)
np.save("afo_singles_percentage_bing.npy", afo_singles_percentage)

#%%    
ofa_singles = []
ofa_singles_percentage = []

for j in range(len(dfa)):
    dfa_train = dfa.copy()
    train = dfa_train.pop(j)
    
    test = pd.concat(dfa_train)
    
    WOB = train['Weight on Bit kkgf'].to_numpy()
    D = train['Diameter mm'].to_numpy()
    N = train['Average Rotary Speed rpm'].to_numpy()
    ROP = train['Rate of Penetration m/h'].to_numpy()

    params = np.ones(2) #initiating the parameters with ones

    model = least_squares(bingham_opt, params,
                          args=(WOB, D, N, ROP)
                          )

    WOB = test['Weight on Bit kkgf'].to_numpy()
    D = test['Diameter mm'].to_numpy()
    N = test['Average Rotary Speed rpm'].to_numpy()
    ROP = test['Rate of Penetration m/h'].to_numpy()
    K = model.x[0]
    a = model.x[1]
    
    local_result = []
    local_result_percentage = []
    
    for j in range(len(ROP)):
        local_result.append(ROP[j] - bingham(K, a, WOB[j], D[j], N[j]))
    
    for j in range(len(ROP)):
        local_result_percentage.append((ROP[j] - bingham(K, a, WOB[j], D[j], N[j]))/ROP[j])

    ofa_singles.append(local_result)
    ofa_singles_percentage.append(local_result_percentage)   
        

np.save("ofa_singles_bing.npy", ofa_singles)
np.save("ofa_singles_percentage_bing.npy", ofa_singles_percentage)  



#%%
