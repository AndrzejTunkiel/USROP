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

results_singles = []

for i in range(len(list(regs))):
    results_singles.append([[],[],[],[],[],[],[]])

results_singles_p = []

for i in range(len(list(regs))):
    results_singles_p.append([[],[],[],[],[],[],[]])

for i in range(len(dfa)):

    for reg in regs:
        
        increment = 577
        depth = increment
        
        while True:
            print(f'''Well: {i}, reg: {reg}: depth: {
                np.round(depth/dfa[i].index.max()*100,1)}%''')
            train = dfa[i][0:depth]
            
            y_train = train['Rate of Penetration m/h'].to_numpy()
            X_train = train.drop(
                labels=['Rate of Penetration m/h'],axis=1).to_numpy()
            
            test = dfa[i][depth:depth+increment]
            
            
            y_test = test['Rate of Penetration m/h'].to_numpy()
            X_test = test.drop(
                labels=['Rate of Penetration m/h'],axis=1).to_numpy()
            
            if depth >= dfa[i].index.max():
                break
            
            regs[reg].fit(X_train, y_train)
            y_pred = regs[reg].predict(X_test)
            
            reg_no = list(regs).index(reg)
            results_singles[reg_no][i].append(y_test - y_pred)
            results_singles_p[reg_no][i].append((y_test - y_pred)/y_test)
            depth = depth + increment
            
np.save("twd.npy", results_singles)
np.save("twd_p.npy", results_singles_p)                
#%%
