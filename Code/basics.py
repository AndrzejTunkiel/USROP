# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob

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
#All for One
X_train = []
X_test = []
y_train = []
y_test = []

for i in range(len(dfa)):

    X_test.append(dfa[i].drop(
        labels=['Rate of Penetration m/h'],axis=1).to_numpy())
    
    y_test.append(
        dfa[i]['Rate of Penetration m/h'].to_numpy())

    dfa_temp = dfa.copy()
    dfa_temp.pop(i)
    X_train.append(pd.concat(dfa_temp))
    y_train.append(X_train[i]['Rate of Penetration m/h'].to_numpy())
    
    X_train[i] = X_train[i].drop(
        labels=['Rate of Penetration m/h'],axis=1).to_numpy()
    

    
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
        #"SVM (SVR)" : SVR()
        }
#%%
afo = pd.DataFrame(columns=['Regressor', 'Well', 'Test score'])

afo_singles = []
for i in range(len(regs)):
    afo_singles.append([])

afo_singles_percentage = []
for i in range(len(regs)):
    afo_singles_percentage.append([])

All_for_one = []
for j in range(len(dfa)):
    test_scores = []
    for reg in regs:
        
        regs[reg].fit(X_train[j], y_train[j])
        
        plt.figure(figsize=(10,10))
        plt.scatter(y_train[j], regs[reg].predict(X_train[j]),s=1, c="black")
        plt.title(f'''Training {reg} for {j}, \nR2 score: 
                  {np.round(regs[reg].score(X_train[j], y_train[j]),3)}''')
        plt.ylim(0,100)
        plt.xlim(0,100)
        plt.plot([0,100],[0,100],c='red')
        plt.show()
    
        plt.figure(figsize=(10,10))
        plt.title(f'''Testing {reg} for {j}, \nR2 score:
                  {np.round(regs[reg].score(X_test[j], y_test[j]),3)}''')
        y_pred = regs[reg].predict(X_test[j])
        plt.scatter(y_test[j], y_pred,s=1, c="green")
        plt.ylim(0,100)
        plt.xlim(0,100)
        plt.plot([0,100],[0,100],c='red')
        plt.show()
        test_scores.append(MAE(y_test[j],y_pred))
        
        reg_no = list(regs).index(reg)
        afo_singles[reg_no].append(y_test[j] - y_pred)
        afo_singles_percentage[reg_no].append((y_test[j] - y_pred)/y_test[j])
        afo = afo.append({'Regressor': reg,
                  'Well' : j,
                  'Test score' : MAE(y_test[j],y_pred)}, ignore_index=True)
    
    print("Test scores:")    
    for i in range(len(test_scores)):
        
        print(f'{list(regs)[i]}: {test_scores[i]}')
        
    All_for_one.append(test_scores)
np.save("afo_singles.npy", afo_singles)
np.save("afo_singles_percentage.npy", afo_singles_percentage)

#%%    
One_for_all = []

ofa_singles = []
for i in range(len(regs)):
    ofa_singles.append([])
    
ofa_singles_percentage = []
for i in range(len(regs)):
    ofa_singles_percentage.append([])

ofa = pd.DataFrame(columns=['Regressor', 'Well', 'Test score'])
for j in range(len(dfa)):
    test_scores = []
    for reg in regs:
        
        regs[reg].fit(X_test[j], y_test[j])
        
        plt.figure(figsize=(10,10))
        plt.scatter(y_test[j], regs[reg].predict(X_test[j]),s=1, c="black")
        plt.title(f'''Training {reg} for {j}, \nR2 score:
                  {np.round(regs[reg].score(X_test[j], y_test[j]),3)}''')
        plt.ylim(0,100)
        plt.xlim(0,100)
        plt.plot([0,100],[0,100],c='red')
        plt.show()
    
        plt.figure(figsize=(10,10))
        plt.title(f'''Testing {reg} for {j}, \nR2 score:
                  {np.round(regs[reg].score(X_train[j], y_train[j]),3)}''')
        y_pred = regs[reg].predict(X_train[j])
        plt.scatter(y_train[j], y_pred,s=1, c="black")
        plt.ylim(0,100)
        plt.xlim(0,100)
        plt.plot([0,100],[0,100],c='red')
        plt.show()
        test_scores.append(MAE(y_train[j],y_pred))
        
        reg_no = list(regs).index(reg)
        ofa_singles[reg_no].append(y_train[j] - y_pred)
        ofa_singles_percentage[reg_no].append((y_train[j] - y_pred)/y_train[j])
        ofa = ofa.append({'Regressor': reg,
                          'Well' : j,
                          'Test score' : MAE(y_train[j],y_pred)},
                         ignore_index=True)
    
    print("Test scores:")    
    for i in range(len(test_scores)):
        
        print(f'{list(regs)[i]}: {test_scores[i]}')
    One_for_all.append(test_scores)
    
np.save('ofa_singles.npy', ofa_singles)    
np.save('ofa_singles_percentage.npy', ofa_singles_percentage)
import seaborn as sns

ax = sns.violinplot(x=ofa['Regressor'],y=ofa['Test score'])
ax.set_ylim([0, 50])
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
plt.title("Predict one well") 
plt.grid()

plt.show()

ax = sns.violinplot(x=afo['Regressor'],y=afo['Test score'])
ax.set_ylim([0, 50])
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
plt.grid()
plt.title("Train on one well") 
plt.show()

#%%
