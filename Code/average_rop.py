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
    
ave_rop = []
for i in dfa:
    print (np.mean(i['Rate of Penetration m/h']))
    ave_rop.append(np.mean(i['Rate of Penetration m/h']))
    
np.save("ave_rop.npy",ave_rop)