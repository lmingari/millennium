import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
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
fname_out       = 'figures/clusters.png'
fname_obs       = config.get(block,'fname_obs')
path            = config.get(block,'path')
debug           = True

###
### Information screen
###
if debug:
    print("""
    ---------------------------------------
    Plot observation site clusters on a map
    ---------------------------------------
    """)

###
### General configuration
###
plt.rcParams.update({'font.size': 6})

###
### Open dataset
###
if debug: print("Opening observation file: {}".format(fname_obs))
fname_obs = join(path,fname_obs)
df = pd.read_csv(fname_obs)

### Plot map
fig = plt.figure()
ax  = fig.add_subplot(1,1,1, projection=crs.PlateCarree())

scatter = ax.scatter(x          = df.longitude, 
                     y          = df.latitude,
                     c          = df.cluster,
                     s          = 12,
                     cmap       = 'Paired',
                     facecolors = 'k',
                     linewidths = 0.2,
                     alpha      = 0.8,
                     transform  = crs.PlateCarree())
gl = ax.gridlines(crs=crs.PlateCarree(),
                  draw_labels = True,
                  linewidth   = 0.5, 
                  color       = 'gray', 
                  alpha       = 0.5, 
                  linestyle   = '--')
ax.add_feature(cfeature.LAND, color="lightgrey", alpha=0.8)
gl.top_labels   = False
gl.right_labels = False
gl.ylabel_style = {'rotation': 90}
ax.legend(*scatter.legend_elements(),title='Clusters')

###
### Output plot
###
if debug: print("Saving output file: {}".format(fname_out))
plt.savefig(fname_out,dpi=200,bbox_inches='tight')
