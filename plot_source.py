import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
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
date1           = config.get(block,'date1')
date2           = config.get(block,'date2')
path1           = config.get(block,'path1')
path2           = config.get(block,'path2')
density         = config.getfloat(block,'dre_density')
fname_plt       = "figures/source.png"
debug           = True

###
### Information screen
###
if debug:
    print(f"""
    ----------------------------------
    Plot emission source term profiles
    ----------------------------------
    Input parameters:
    DRE density = {density} kg/m3
    """)

times = [datetime.strptime(item.split('-')[0], '%Y%m%d') for item in [date1,date2]]

fmt = "%d %B %Y at %H UTC"
plot_conf = [
        {'date': date1,
         'path': path1,
         'time': times[0].strftime(fmt),
         'title': '(a) Comendite phase'
         },
        {'date': date2,
         'path': path2,
         'time': times[1].strftime(fmt),
         'title': '(b) Trachyte phase'
         },
        ]
nax = len(plot_conf)
###
### Plot emission source profiles
###
fig, axs = plt.subplots(nax,figsize=(10,6*nax))

for ax, item in zip(axs,plot_conf):
    date      = item['date']
    path      = item['path']
    time      = item['time']
    title     = item['title']
    ###
    ### Open ensemble of emission source terms
    ###
    fname_src = join('OUTPUT',date,'millennium.src.nc')
    if debug: print(f"Opening emisison source file: {fname_src}")
    ds = xr.open_dataset(fname_src)
    ###
    ### Open factor weights
    ###
    fname_w = join(path,f"factors_{date}.dat")
    if debug: print(f"Opening weight factors file: {fname_w}")
    w  = np.loadtxt(fname_w)
    ds['w'] = ('ens',w)
    #
    X = ds.time.values / 3600.0 # time in h
    Z = ds.lev.values  / 1000.0 # hight asl in km
    DT = (X[1]-X[0])*3600.      # time interval in sec
    #
    C = ds.src.dot(ds.w).values / 1000.0
    M = ds.mfr.dot(ds.w).values
    #
    im = ax.pcolormesh(X,Z,C.T,
            shading = 'gouraud',
            cmap    = 'YlOrRd')
    cbar = fig.colorbar(im,
            orientation="horizontal", 
            label = r'Linear source emission strength ($\times 10^3$) [$kg~s^{-1}~m^{-1}$]',
            ax=ax)
    ax2 = ax.twinx()
    l2, = ax2.plot(X,1E-7*M,
            label     = "Emission rate", 
            color     = "blue",
            linestyle = "dashdot")
    ax3 = ax.twinx()
    l3, = ax3.plot(X,DT*1E-9*M.cumsum()/density,
            label     = "Cumulative Erupted Volume",
            color     = "black",
            linestyle = "solid")
    ###
    ### Configure plots
    ###
    ax.set(
        ylabel = 'Altitude [km asl]',
        xlabel = f'Simulation time [hours since {time}]',
        title  = title,
        ylim   = [0,46],
        )
    ax2.set(
        ylabel = r'Emission rate ($\times 10^7$) [kg/s]',
        )
    ax2.spines['left'].set_position(('outward', 50))
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.set_ticks_position('left')
    ax3.set(
        ylabel = r'Cumulative Erupted Volume [$km^3$ DRE]',
        ylim   = [0,10],
        )
    #
    plt.legend(
        handles=[l2,l3], 
        loc='lower right',
        bbox_to_anchor=(0.98, 0.1)
        )
    #
    erupted = DT*1E-9*M.cumsum()/density
    print("Total volume erupted:")
    print(erupted.max())
    #
    ds.close()

fig.tight_layout()
if debug: print(f"Saving plot: {fname_plt}")
fig.savefig(fname_plt,
            dpi=200,
            bbox_inches='tight')
