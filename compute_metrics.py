import pandas as pd
import numpy as np
import xarray as xr
from os.path import join
from configparser import ConfigParser
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

###
### Read configuration file
###
config = ConfigParser(inline_comment_prefixes="#")
config.read('config.ini')

####
#### Parameters
####
block           = 'TOTAL'
path_obs        = config.get('DATA','path')
fname_obs       = config.get(block,'fname_obs')
path1           = config.get(block,'path1')
path2           = config.get(block,'path2')
debug           = True

###
### Information screen
###
if debug:
    print("""
    --------------------------
    Compute validation metrics
    --------------------------
    """)

data = []

dates = [
    "19620125-19620129",
    "19680930-19681004",
    "19710409-19710413",
    "19760229-19760304",
    "19771021-19771025",
    "19921030-19921103"
    ]

###
### Read obs data
###
fname_obs = join(path_obs,fname_obs)
if debug: print("Opening observation file: {}".format(fname_obs))
df = pd.read_csv(fname_obs)
nobs = len(df)
if debug: print("Number of observations: {}".format(nobs))

for date1 in dates:
    ###
    ### Read model data
    ###
    fname_an = join(path1,f"analysis_{date1}.nc")
    if debug: print(f"Opening analysis file: {fname_an}")
    ds1 = xr.open_dataset(fname_an)
    for date2 in dates:
        fname_an = join(path2,f"analysis_{date2}.nc")
        if debug: print(f"Opening analysis file: {fname_an}")
        ds2 = xr.open_dataset(fname_an)
        x = ds1.analysis + ds2.analysis
        ###
        ### Interpolation to observation sites
        ###
        if debug: print("Performing interpolations")
        lat_obs = xr.DataArray(df['latitude'], dims='loc')
        lon_obs = xr.DataArray(df['longitude'],dims='loc')
        y       = x.interp(lat=lat_obs,lon=lon_obs)
        df['analysis'] = y
        ###
        ### Compute accuracy metrics
        ###
        yo = df['thickness']
        ye = df['error']
        ym = df['analysis']
        #
        e     = (yo-ym)/ye
        bias  = e.mean()
        mae   = e.abs().mean()
        mse   = (e**2).mean()
        #
        e1    = (yo-ym).abs()
        e2    = yo.abs()+ym.abs()
        smape = (e1[e2>0]/e2[e2>0]).mean()
        #
        mare  = ((yo-ym).abs()/yo).mean()
        #
        #seek 68% ratio
        ratio, rstep = 1.0, 1E-2
        for istep in range(1000):
            ratio += rstep
            hits  = ((ym<ratio*yo) & (ratio*ym>yo)).sum()/len(ym)
            if hits>0.68: break
        #
        data.append({
            'date_comendite': date1,
            'date_trachyte':  date2,
            'rmse':           np.sqrt(mse),
            'bias':           bias,
            'mae':            mae,
            'smape':          100*smape,
            'mare':           100*mare,
            'hits':           100*hits,
            'ratio':          ratio,
            })
        ds2.close()
    ds1.close()

fname_csv = "metrics2.csv"
if debug: print("Saving output file: {}".format(fname_csv))
df_out = pd.DataFrame(data)
df_out.to_csv(fname_csv)

#print(df_out[df_out.rmse==df_out.rmse.min()])
