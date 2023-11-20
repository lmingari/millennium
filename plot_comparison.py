import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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
block           = "TOTAL"
path_obs        = config.get('DATA','path')
fname_obs       = config.get(block,'fname_obs')
date1           = config.get(block,'date1')
date2           = config.get(block,'date2')
path1           = config.get(block,'path1')
path2           = config.get(block,'path2')
fname_plt       = "figures/comparison.png"
xmin,xmax       = 1E-4,1E3
debug           = True

###
### Plot model vs observations
###
fig, axs = plt.subplots(3,figsize=(8,14))

####
#### Read obs data
####
fname_obs = join(path_obs,fname_obs)
if debug: print("Opening observation file: {}".format(fname_obs))
df = pd.read_csv(fname_obs)
nobs = len(df)
if debug: print("Number of observations: {}".format(nobs))

###
### Read model data
###
fname_an = join(path1,f"analysis_{date1}.nc")
if debug: print("Opening analysis file: {}".format(fname_an))
ds1 = xr.open_dataset(fname_an)
fname_an = join(path2,f"analysis_{date2}.nc")
if debug: print("Opening analysis file: {}".format(fname_an))
ds2 = xr.open_dataset(fname_an)
###
### Interpolation to observation sites
###
if debug: print("Performing interpolations")
lat_obs = xr.DataArray(df['latitude'], dims='loc')
lon_obs = xr.DataArray(df['longitude'],dims='loc')
y1      = ds1.interp(lat=lat_obs,lon=lon_obs)
y2      = ds2.interp(lat=lat_obs,lon=lon_obs)
df['analysis1'] = y1.analysis
df['analysis2'] = y2.analysis
df['analysis']  = df.analysis1 + df.analysis2

###
### Plot 
###
plot_conf = [
        {'analysis':'analysis1',
         'observation': 'comendite',
         'dataset': 'Assimilation dataset\n(23 points)',
         'title': '(a) Comendite phase'},
        {'analysis':'analysis2',
         'observation': 'trachyte',
         'dataset': 'Assimilation dataset\n(23 points)',
         'title': '(b) Trachyte phase'},
        {'analysis':'analysis',
         'observation': 'thickness',
         'dataset': 'Validation dataset\n(83 points)',
         'title': '(c) Total deposit'},
        ]

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
for ax,item in zip(axs,plot_conf):
    analysis    = item['analysis']
    observation = item['observation']
    title       = item['title']
    ax.plot(df[analysis],df[observation],
            marker = 'o',
            ls     = 'none',
            color  = 'tab:red',
            alpha  = 0.7,
            )
    ax.plot([xmin,xmax],[xmin,xmax],        'k-',  lw = 0.8, label="Ideal")
    ax.plot([xmin,xmax],[10*xmin,10*xmax],  'k--', lw = 0.8, label="1:10 ratio")
    ax.plot([xmin,xmax],[0.1*xmin,0.1*xmax],'k--', lw = 0.8)

    yo = df.loc[~df[observation].isna(),observation]
    ym = df.loc[~df[observation].isna(),analysis]
    #seek 68% ratio
    ratio, rstep = 1.0, 1E-2
    for istep in range(1000):
        ratio += rstep
        hits  = ((ym<ratio*yo) & (ratio*ym>yo)).sum()/len(ym)
        if hits>0.68: break
    print( f'ratio={ratio}' )
    label = f'1:{ratio:.2f} ratio (68% data)'
    ax.plot([xmin,xmax],[ratio*xmin,ratio*xmax], 'b--', lw = 0.8, label=label)
    ax.plot([xmin,xmax],[xmin/ratio,xmax/ratio], 'b--', lw = 0.8)
    ax.text(0.04, 0.96, item['dataset'], 
            transform=ax.transAxes, 
            fontsize=10,
            verticalalignment='top', 
            bbox=props)
    ax.grid()
    ax.set(ylabel = 'Deposit thickness - Observation [cm]',
           xlabel = 'Deposit thickness - Analysis [cm]',
           xscale = 'log',
           yscale = 'log',
           xlim   = (xmin,xmax),
           ylim   = (xmin,xmax),
           title  = title
           )
    ax.legend(loc=4)
    ax.label_outer()

#
# Show outliers
#
#for ip in [0,1]:
#    item        = plot_conf[ip]
#    analysis    = item['analysis']
#    observation = item['observation']
#    title       = item['title']
#    print(f"**** Outliers {analysis}")
#    subset = df.loc[df[observation]>=0]
#    print(subset.loc[(subset[analysis]<xmin) | (subset[observation]<xmin)])

fig.tight_layout()
fig.savefig(fname_plt,
            dpi=200,
            bbox_inches='tight')
