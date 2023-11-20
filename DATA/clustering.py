import pandas as pd
import numpy as np
from metrics import get_affinity
from sklearn.cluster import SpectralClustering
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
n_clusters      = config.getint(block,'n_clusters')
thickness_min   = config.getfloat(block,'thickness_min')
debug           = True

###
### Information screen
###
if debug:
    print("""
    --------------------------------------------------
    Apply spectral clustering to an observation dataset
    --------------------------------------------------
    Input parameters:
    number of clusters = {n_clusters}
    minimum thickness = {thickness_min} cm
    """.format(n_clusters = n_clusters,
               thickness_min = thickness_min,
               )
          )

### Read observations
df = pd.read_csv('dataset.csv', index_col=0)

#Compute similarity matrix
A = get_affinity(df, thickness_min)

#Clustering
model = SpectralClustering(
        n_clusters=n_clusters,
        affinity = 'precomputed',
        )
labels = model.fit_predict(A)
df['cluster'] = labels
df.sort_values('cluster',inplace=True)
df.set_index('cluster', inplace=True)

#Output dataset
if debug: print("Saving output file: {}".format(fname_clusters))
column_list = ["latitude","longitude","thickness","comendite","trachyte"]
df.to_csv(fname_clusters,columns=column_list)
