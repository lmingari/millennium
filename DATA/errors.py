import numpy as np
import pandas as pd
from metrics import get_affinity
from configparser import ConfigParser

###
### Read configuration file
###
config = ConfigParser(inline_comment_prefixes="#")
config.read('../config.ini')

###
### Parameters
###
block           = 'DATA'
fname_clusters  = config.get(block,'fname_clusters')
fname_obs       = config.get(block,'fname_obs')
thickness_min   = config.getfloat(block,'thickness_min')
relative_error  = config.getfloat(block,'relative_error')
debug           = True

###
### Information screen
###
if debug:
    print("""
    --------------------------------------------------
    Add errors to an observation dataset with clusters
    --------------------------------------------------
    Input parameters:
    minimum thickness = {thickness_min} cm
    default relative error = {relative_error} %
    """.format(thickness_min = thickness_min,
               relative_error = 100*relative_error,
               )
          )

### Open dataset
if debug: print("Opening observation file: {}".format(fname_clusters))
df = pd.read_csv(fname_clusters)
nobs = len(df)

### Compute errors
if debug: print("Computing errors")
minimum_error = thickness_min * relative_error
df['true']  = df.groupby("cluster")["thickness"].transform('mean')
df['error'] = df.groupby("cluster")["thickness"].transform('std')
df['error'].where(df.error>0.0,relative_error*df.true, inplace=True)
df['error'].where(df.error>minimum_error,minimum_error, inplace=True)

###
### Output table
###
if debug: print("Saving output file: {}".format(fname_obs))
df.to_csv(fname_obs,
          columns=["latitude",
                   "longitude",
                   "thickness",
                   "comendite",
                   "trachyte",
                   "error",
                   "cluster",
                   ],
          index=False)
