import numpy as np
from assimilation import GNC as AssimilationMethod
from os.path import join
from configparser import ConfigParser

###
### Read configuration file
###
config = ConfigParser(inline_comment_prefixes="#")
config.read('config.ini')

###
### Parameters
###
block           = 'COMENDITE'
path_obs        = config.get('DATA','path')
fname_obs       = config.get(block,'fname_obs')
fname_ens       = config.get(block,'fname_ens')
path            = config.get(block,'path')
dataset         = config.get(block,'dataset')
bulk_density    = config.getfloat(block,'bulk_density')
max_iterations  = config.getint(block,'max_iterations')
debug           = True

###
### Information screen
###
if debug:
    print(f"""
    ---------------------------------
    Assimilation using the GNC method
    ---------------------------------
    Input parameters:
    bulk density = {bulk_density} kg/m3
    maximum number of iterations = {max_iterations}
    dataset {dataset}
    """)

###
### Use the GNC method
###
data = AssimilationMethod(max_iterations)

####
#### Read obs data
####
fname_obs = join(path_obs,fname_obs)
if debug: print(f"Opening observation file: {fname_obs}")
data.read_observations(fname_obs,dataset)
if debug: print(f"Number of observations: {data.nobs}")

dates = [
    "19620125-19620129",
    "19680930-19681004",
    "19710409-19710413",
    "19760229-19760304",
    "19771021-19771025",
    "19921030-19921103"
    ]
for date in dates:
    ###
    ### Read model data
    ###
    fname = join("OUTPUT",date,fname_ens)
    if debug: print(f"Opening simulation output file: {fname}")
    data.read_ensemble(fname,bulk_density)

    ###
    ### Interpolation to observation sites
    ###
    if debug: print("Performing interpolations")
    data.apply_ObsOp()

    ###
    ### Assimilate
    ###
    if debug: print("Assimilating data")
    w = data.assimilate()

    ###
    ### Save analysis data
    ###
    fname_an = f"analysis_{date}.nc"
    fname_an = join(path,fname_an)
    if debug: print(f"Saving analysis output file: {fname_an}")
    data.to_netcdf(fname_an)

    ###
    ### Save weight factors
    ###
    fname_w = join(path,f"factors_{date}.dat")
    if debug: print(f"Saving weight factors file: {fname_w}")
    with open(fname_w,'w') as f:
        for item in w:
            f.write(f"{item:.9E}\n")
