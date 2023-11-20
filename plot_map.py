import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as crs
import cartopy.feature as cfeature
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
block           = 'TOTAL'
fname_obs       = config.get(block,'fname_obs')
date1           = config.get(block,'date1')
date2           = config.get(block,'date2')
path1           = config.get(block,'path1')
path2           = config.get(block,'path2')
levels          = config.get(block,'levels')
fname_plt       = "figures/map.png"
plot_obs        = False
debug           = True

###
### Information screen
###
if debug:
    print("""
    -------------------------------
    Plot analysis contours on a map
    -------------------------------
    """)

###
### General configuration
###
#plt.rcParams.update({'font.size': 6})

###
### Open observation dataset
###
#if plot_obs:
#    if debug: print("Opening observation file: {}".format(fname_obs))
#    df = pd.read_csv(fname_obs)
#    dataset = 'assimilation'
#    plot_conf = {
#        'validation':   {'marker' : '^',
#                         'color'  : 'g',
#                         'label'  : 'Validation dataset'},
#        'assimilation': {'marker' : '.',
#                         'color'  : 'k',
#                         'label'  : 'Assimilation dataset'}
#        }

###
### Read analysis file
###
fname_an = join(path1,f"analysis_{date1}.nc")
if debug: print(f"Opening analysis file: {fname_an}")
ds1 = xr.open_dataset(fname_an)
fname_an = join(path2,f"analysis_{date2}.nc")
if debug: print(f"Opening analysis file: {fname_an}")
ds2 = xr.open_dataset(fname_an)

### Plot map
fig, axs = plt.subplots(nrows=3,
                        subplot_kw={'projection': crs.PlateCarree()},
                        figsize=(8,15)
                        )

levels = [float(item) for item in levels.split()]
cmap   = plt.cm.RdYlBu_r
norm   = BoundaryNorm(levels,cmap.N)

###
### Comendite deposit
###
axs[0].contourf(ds1.lon,ds1.lat,10.0*ds1.analysis,
                 levels = levels,
                 norm = norm,
                 cmap = cmap,
                 extend='max',
                 transform = crs.PlateCarree()
                 )
###
### Trachyte deposit
###
axs[1].contourf(ds2.lon,ds2.lat,10.0*ds2.analysis,
                 levels = levels,
                 norm = norm,
                 cmap = cmap,
                 extend='max',
                 transform = crs.PlateCarree()
                 )
###
### Total deposit
###
x = 10.0*(ds1.analysis+ds2.analysis)
fc = axs[2].contourf(x.lon,x.lat,x,
                 levels = levels,
                 norm = norm,
                 cmap = cmap,
                 extend='max',
                 transform = crs.PlateCarree()
                 )

# Adjust the location of the subplots on the page to make room for the colorbar
fig.subplots_adjust(bottom=0.1,
                    top=0.95,
                    left=0.05,
                    right=0.95,
                    hspace=0.15)

# Add a colorbar axis at the bottom of the graph
cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])

# Draw the colorbar
cbar=fig.colorbar(fc, 
                  ticks       = levels,
                  orientation = 'horizontal',
                  label       = 'Deposit thickness in mm',
                  cax         = cbar_ax,
                  )

BORDERS = cfeature.NaturalEarthFeature(scale='50m',
                                       category='cultural',
                                       name='admin_0_countries',
                                       edgecolor='k',
                                       facecolor='none')
LAND = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                    edgecolor='none',
                                    facecolor='lightgrey')

for ax in axs:
    ax.add_feature(BORDERS, linewidth=0.4)
    ax.add_feature(LAND,zorder=0)
    gl = ax.gridlines(crs=crs.PlateCarree(),
                      draw_labels = True,
                      linewidth   = 0.5, 
                      color       = 'gray', 
                      alpha       = 0.5, 
                      linestyle   = '--')
    gl.top_labels   = False
    gl.right_labels = False
    gl.ylabel_style = {'rotation': 90}

axs[0].set_title("(a) Comendite phase")
axs[1].set_title("(b) Trachyte phase")
axs[2].set_title("(c) Total deposit")

###
### Plot sampling site locations
###
#if plot_obs:
#    ax.scatter(x          = df.loc[df.dataset==dataset,'longitude'], 
#               y          = df.loc[df.dataset==dataset,'latitude'],
#               marker     = plot_conf[dataset]['marker'],
#               s          = 12,
#               edgecolors = plot_conf[dataset]['color'],
#               facecolors = 'None',
#               linewidths = 0.6,
#               alpha      = 0.8,
#               label      = plot_conf[dataset]['label'],
#               transform  = crs.PlateCarree())
#    ax.legend()

###
### Output plot
###
#fig.tight_layout()
if debug: print("Saving output file: {}".format(fname_plt))
plt.savefig(fname_plt,dpi=200,bbox_inches='tight')
