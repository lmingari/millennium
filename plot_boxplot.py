import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
block           = 'DATA'
fname_out       = 'figures/boxplot.png'
fname_obs       = config.get(block,'fname_obs')
path            = config.get(block,'path')
debug           = True

###
### Information screen
###
if debug:
    print("""
    ---------------------------------------
    Plot boxplot from a clusterised dataset
    ---------------------------------------
    """)

###
### Open dataset
###
if debug: print("Opening observation file: {}".format(fname_obs))
fname_obs = join(path,fname_obs)
df = pd.read_csv(fname_obs)
nc = df.groupby("cluster").ngroups

###
### Plot boxes
###
fig, ax = plt.subplots()
df.boxplot(positions = np.arange(nc),
           column    = ['thickness'],
           by        = 'cluster',
           whis      = (0, 100),
           ax        = ax)
df.plot.scatter(x     = 'cluster',
                y     = 'thickness',
                color = 'red',
                alpha = 0.5,
                label = 'Measurements',
                ax    = ax)
ax.set(xlabel = 'Cluster label',
       ylabel = 'Deposit thickness [cm]',
       title  = 'Clustered box plot diagram',
       yscale = 'log',
       )
ax.grid(axis='y', color = 'gray', linestyle='--', linewidth=0.4)
fig.suptitle("")

###
### Output plot
###
if debug: print("Saving output file: {}".format(fname_out))
fig.savefig(fname_out,dpi=200,bbox_inches='tight')
