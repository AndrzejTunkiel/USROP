import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob

np.random.seed(0)
#%%
#Get list of files from directory
filelist = glob(r'Norway*.csv', recursive=False)

print ("Detected logs:\n")

filelist.sort()

for i in range(len(filelist)):
    print ('[' + str(i) + ']' + " " + filelist[i].split('\\')[-1] + " " +
           str(os.path.getsize(filelist[i])//1000000) + 'MB')

print ()

#%%
# import csvs to a list of dataframes
df = []

for i in filelist:
    df.append(pd.read_csv(i))
    
    

#%%
#debug cell, find attributes
searchphrase = "gamma"
searchphrase = searchphrase.lower()
for j in range(len(df)):
    print("")
    print(j)
    print(f'{filelist[j]}')
    for i in list(df[j]):
        if (i.lower().find(searchphrase) > -1):
            df[j][i].ffill().plot()
            df[j]['Rate of Penetration m/h'].ffill().plot()
            plt.ylim(0,500)
            plt.title(f'From well {j}')
            plt.legend()
            print (i)
            plt.show()
#%%


df[0]["USROP Gamma gAPI"] = df[0]['MWD Gamma Ray (API BH corrected) gAPI']


df[1]["USROP Gamma gAPI"] = df[1]['MWD Gamma Ray (API BH corrected) gAPI']

df[2]["USROP Gamma gAPI partial"] = df[2].iloc[:40000]['MWD Gamma Ray (API BH corrected) gAPI'].combine_first(df[2]['Gamma Ray, Average gAPI']).astype(float)
df[2]["USROP Gamma gAPI"] = df[2]['USROP Gamma gAPI partial'].combine_first(df[2]['ARC Gamma Ray (BH corrected) gAPI']).astype(float)


df[3]["USROP Gamma gAPI"] = df[3]['ARC Gamma Ray (BH corrected) gAPI'].combine_first(df[3]['Gamma Ray, Average gAPI']).astype(float)

df[4]["USROP Gamma gAPI"] = df[4]['ARC Gamma Ray (BH corrected) gAPI'].combine_first(df[4]['Gamma Ray, Average gAPI']).astype(float)

df[5]["USROP Gamma gAPI"] = df[5]['Gamma Ray, Average gAPI']

df[6]["USROP Gamma gAPI"] = df[6]['MWD Gamma Ray (API BH corrected) gAPI']

            
#%%
# debug cell, find attributes shared by all files
for i in list(df[0]):
    c = 0
    for j in range(1,len(df)):
        for k in list(df[j]):
            if (k.find(i) > -1):
                c = c + 1
    if c >= 6: print(f"All have {i}")

            
#%%
#select attributes

names = ['Measured Depth m','Weight on Bit kkgf',
         'Average Standpipe Pressure kPa', 'Average Surface Torque kN.m', 
         'Rate of Penetration m/h','Average Rotary Speed rpm',
         'Mud Flow In L/min', 'Mud Density In g/cm3', 'name',
         'Average Hookload kkgf',"Hole Depth (TVD) m","USROP Gamma gAPI"]

#%%
#create clean dataframes, with just the selected attributes
df_clean = []

for i in range(len(df)):
    dftemp = pd.DataFrame()
    for j in names:
        dftemp[j] = df[i][j]
    df_clean.append(dftemp)
    
#%%
#debug cell plot some charts
# =============================================================================
# 
# for i in range(len(df_clean)):
#     for j in names:
#         x = df_clean[i]['Measured Depth m'].ffill()
#         y = df_clean[i][j].ffill()
#         plt.scatter(x,y, s=1, label=j)
#     plt.legend()
#     plt.show()
# =============================================================================
#%% Cleaning up rows without explicit ROP


#%%
# debug cell, plot some charts
for i in range(len(df_clean)):
    for j in list(df_clean[i]):

        x = df_clean[i]['Measured Depth m'].ffill()
        y = df_clean[i][j].ffill()
        plt.plot(x,y, label=j)
        plt.title(f'Well {i}')
        plt.grid()
        plt.legend(loc=1)
        plt.show()

    
#%%
#manual filtering of outliers

for i in range(len(df_clean)):
    df_clean[i] = df_clean[i][~df_clean[i].name.str.contains('36 in.')]
    df_clean[i] = df_clean[i][~df_clean[i].name.str.contains('26in')]
    df_clean[i] = df_clean[i][~df_clean[i].name.str.contains('36in')]


for i in range(len(df_clean)):
    df_clean[i] = df_clean[i].ffill().bfill()
    
def remove_below(df,name,value):
    df = df[df[name] > value]
    return df

def remove_above(df,name,value):
    df = df[df[name] < value]
    return df

for i in range(len(df_clean)):

    df_clean[i] = remove_below(df_clean[i], 'Weight on Bit kkgf', 0)
    df_clean[i] = remove_above(df_clean[i], 'Weight on Bit kkgf', 35)
    df_clean[i] = remove_below(df_clean[i], 'Mud Density In g/cm3', 0)
    df_clean[i] = remove_below(df_clean[i], 'Mud Flow In L/min', 0)
    df_clean[i] = remove_below(df_clean[i], 'Average Surface Torque kN.m', 0)
    df_clean[i] = remove_above(df_clean[i], 'Rate of Penetration m/h', 100)
    df_clean[i] = remove_above(df_clean[i], 'Average Standpipe Pressure kPa',
                               25000)

df_clean[1] = remove_below(df_clean[6], 'Measured Depth m', 300)
df_clean[6] = remove_below(df_clean[6], 'Measured Depth m', 225)
df_clean[4] = remove_below(df_clean[4], 'Measured Depth m', 1400)
#%%
# convert strings to well diameters
sizeconvert = {
r'8.5 in Section - MD Log'  : 215.9,
r'12.25 in Section - MD Log' : 311.15,
r'Real Time MWD/LWD Mudlog Data - 17.5 in. section - MD Log' : 444.5,
r'8 1/2 in Section - MD Log'  : 215.9,
r'17 1/2in Section. - MD Log': 444.5,
r'12 1/4in Section - MD Log' : 311.15,
r'8.5 in - MD Log'  : 215.9,
r'17.5in Section - MD Log': 444.5,
r'8.5in Section - MD Log'   : 215.9,
r'12.25in. section - MD Log' : 311.15,
r'Real Time SLB &amp; Geoservices data - 8.5in. Section - MD Log' : 215.9,
r'17.5 in section Combined SLB and Geoservices Data - MD Log': 444.5}


for i in range(len(df_clean)):
    df_clean[i].replace(sizeconvert, inplace=True)
    df_clean[i] = df_clean[i].rename(columns={"name": "Diameter mm"})

#%%

#Removing areas of overlapping wells of different size
for i in range(20):
    for i in range(len(df_clean)):
        df_clean[i] = df_clean[i].sort_values(by="Measured Depth m")
        df_clean[i]["diameter shake"] = df_clean[i]["Diameter mm"].diff()
        
        df_clean[i]["Diameter mm"].plot()
        plt.show()
        df_clean[i] = df_clean[i][df_clean[i]["diameter shake"] == 0 ]
        df_clean[i] = df_clean[i].drop(["diameter shake"], axis=1)

#%% balancing dataset in terms of samples per meter

spml_array = np.zeros(len(df_clean)) #samples per measured length
ml_array = np.zeros(len(df_clean)) 
for i, val in enumerate(df_clean):
    print (f'Well {i}')
    print (f'Sample count = {len(val)}')
    ml = np.max(val['Measured Depth m']) - np.min(val['Measured Depth m'])
    print (f'Total measured length = {ml}')
    print(f'Samples per meter = {len(val)/ml}')
    print()
    spml_array[i] = len(val)/ml
    ml_array[i] = ml

min_spml = np.min(spml_array)
#%%
for i in range(len(df_clean)):
    df_clean[i] = df_clean[i].sample(n=int(min_spml*ml_array[i]),
                                     random_state=42)

#%%
#debug cell, used in converting to well diameter
# =============================================================================
# 
# for i in df_clean:
#     print(i['name'].unique())
# =============================================================================



#%%
    
names = ["USROP_A 0 N-NA_F-9_Ad.csv",
         "USROP_A 1 N-S_F-7d.csv",
         "USROP_A 2 N-SH_F-14d.csv",
         "USROP_A 3 N-SH-F-15d.csv",
         "USROP_A 4 N-SH_F-15Sd.csv",
         "USROP_A 5 N-SH-F-5d.csv",
         "USROP_A 6 N-SH_F-9d.csv"]

for i in range(len(df_clean)):
    print(i)
    print(np.min(df_clean[i]['Measured Depth m']))
    print(np.max(df_clean[i]['Measured Depth m']))
    print(f'''Delta: {(np.max(df_clean[i]["Measured Depth m"])) -
          (np.min(df_clean[i]["Measured Depth m"])) }''')
    print(len(df_clean[i]))
    df_clean[i] = df_clean[i].sort_values(by = ['Measured Depth m'],
                                          ignore_index=True)
    df_clean[i]['Measured Depth m'].plot()
    plt.show()
    df_clean[i].to_csv(names[i])
    
#%%

description = pd.DataFrame(columns=['File Name',
                                    'Starting Measured Depth [m]',
                                    'Final Measured Depth [m]',
                                    'Available length [m]',
                                    'Sample count'])

for i, val in enumerate(df_clean):
    description = description.append({
        'File Name' : names[i],
        'Starting Measured Depth [m]' : np.round(np.min(val['Measured Depth m']),0),
        'Final Measured Depth [m]' :  np.round(np.max(val['Measured Depth m']),0),
        'Available length [m]' :  np.round(np.max(val['Measured Depth m']) - 
            np.min(val['Measured Depth m']),0),
       'Sample count' : len(val)
        }, ignore_index=True)
    
description.to_csv('Description.csv')