# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:27:11 2020

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

labels = ['Measured\nDepth\n[m]',
'Weight\non Bit\n[kkgf]',
'Average\nStandpipe\nPressure\n[kPa]',
'Average\nSurface\nTorque\n[kN.]m',
'Rate of\nPenetration\n[m/h]',
'Average\nRotary\nSpeed\n[rpm]',
'Mud\nFlow In\n[L/min]',
'Mud\nDensity In\n[g/cm3]',
'Diameter\n[mm]',
'Average\nHookload\n[kkgf]',
'Hole Depth\n(TVD) [m]',
'USROP\nGamma\n[gAPI]']

for i in range(len(list(dfa[0]))):
    fig, axs = plt.subplots(1, len(list(dfa[i]))-1, figsize=(10,15), sharey=True)
    for j, val in enumerate(dfa[i]):
        if j ==0:
            continue
        

        j = j - 1
        print(val)
        axs[j].plot(dfa[i][val], np.linspace(0,1,len(dfa[i][val])))
        
        axs[j].set_xlabel(labels[j], rotation=0)
        
        axs[j].invert_yaxis()
        #axs[j].tick_params(labelrotation=90)
        axs[j].set_ylim(1,0)
        
        for tick in axs[j].get_xticklabels():
            tick.set_rotation(90)
        
    
    plt.tight_layout()
    plt.savefig(f'Welldata for well {i}.pdf')
    plt.show()
    
    
#%%
total_length = 0
total_samples = 0
for i in range(len(dfa)):
    print(i)
    print(np.min(dfa[i]['Measured Depth m']))
    print(np.max(dfa[i]['Measured Depth m']))
    print(f'''Delta: {(np.max(dfa[i]["Measured Depth m"])) -
          (np.min(dfa[i]["Measured Depth m"])) }''')
    total_length = total_length + (np.max(dfa[i]["Measured Depth m"])) - (np.min(dfa[i]["Measured Depth m"])) 
    total_samples = total_samples + len(dfa[i])
    print(len(dfa[i]))

print(f'total samples: {total_samples}')
print(f'total length: {total_length}')