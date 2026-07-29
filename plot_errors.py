

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmocean as cmo
import matplotlib.dates as mdates
import os 

############################## OPEN MODEL DATA ##############################

# Get a list of netcdfs in the folder
directory = "/global/scratch/users/siennaw/scripts/enkf_sediment" 



if 1 : 


    csv_files = [
        os.path.join('metrics/', f) 
        for f in os.listdir('metrics/') 
        if f.endswith(".csv")
    ]

    fig = plt.figure(figsize=(11,6))
    ax = plt.gca()
    for fn in csv_files:
        df = pd.read_csv(fn)
        df = df.dropna(how='any')
        ax.plot(df.rmse, '-', label=fn, linewidth=2, alpha=0.8) 

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_ylim(0, 25)
    ax.set_xlim(0, 200 ) #4000)
    ax.grid(alpha=0.2)
    plt.tight_layout() 
    fig.savefig('errors.png')

nc_files = [ f
    for f in os.listdir(directory) 
    if f.endswith(".nc")
]


if 0 : 

    fns = nc_files #["flocmod_38.nc", "flocmod_39.nc", "flocmod_40.nc"]



    for fn in fns: 
        print(fn)

        try:
            ds = xr.open_dataset(fn, decode_times=False)
        except:
            print("delete %s" % fn) 
            continue  
        # volume = 4/3 * np.pi * (ds.Ds/2)**3 
        ds = ds.dropna(dim="N", how="any")



        lstdata= pd.read_csv("lisst_data.csv")

        seconds = lstdata['seconds'].values

        size_classes = lstdata.columns[1:]

        df = pd.DataFrame(index=seconds)
        df['rmse'] = np.nan
        df['skill'] = np.nan


        fig = plt.figure(figsize=(8, 4))
        ax = plt.gca()


        for i,t in enumerate(seconds):
            conc = ds.ssc.sel(time=t, method='nearest')
            conc = conc @ ds.H 
            model = conc.mean(dim='N')
            obs = lstdata.iloc[i, 1:].values
            rmse = np.mean(np.sqrt((model - obs)**2)) 
            # print('at time %d rmse = %2.1f' % (t, rmse))
            # 1. Standard Error of the Mean (SEM)
            # Measures the precision of the sample mean estimate (std / sqrt(N))
            se_obs = np.std(obs, ddof=1) / np.sqrt(len(obs))
            se_pred = np.std(model, ddof=1) / np.sqrt(len(model))

            # ax.plot(t, rmse, 'o', color=color, markersize=5)

            baseline = np.full_like(obs, np.mean(obs))  # Default: sample mean
            rmse_pred = np.mean((model - obs) ** 2)
            mse_ref = np.mean((baseline - obs) ** 2)
        
            skill_score = 1 - (rmse_pred / mse_ref)

            df['skill'].loc[t] = skill_score
            df['rmse'].loc[t] = rmse 


        df = df.replace([np.inf, -np.inf], np.nan)
        df.dropna(how='all', inplace=True)
        df.loc[df.skill<-10] = np.nan
        print("MEAN RMSE: ",  np.nanmean(df.rmse))
        print("MEAN SKILL: ",  np.nanmean(df.skill))
        df.to_csv("metrics/errors_%s.csv" % fn.replace('.nc', ''))

        # l = lstdata.iloc[0, 1:].values
        # # Hi = np.linalg.inv(ds.H.values)
        # H = ds.H.values 
        # H[H > 0] = 1/H[H > 0]
        # # print(H)
        # ic = H @ l 
        # ic = ic.astype(int)

# Look @ metrics

