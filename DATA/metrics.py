import numpy as np
import math
from sklearn.metrics import pairwise_distances

################# Inputs ################# 
deltas        = [150.0,1.0]               # Scales for distances [km,dimensionless]
########################################## 

def get_affinity(df,thickness_min):
    #Define X (n_samples_X, n_features)
    X = df[['latitude','longitude']].to_numpy()

    #Compute distance matrix
    D1 = pairwise_distances(X,metric=haversine_distance)

    #Define X (n_samples_X, n_features)
    X = df[['thickness']].to_numpy()
    X[X<thickness_min] = thickness_min

    #Compute distance matrix
    D2 = pairwise_distances(X,metric=log_distance)

    #Compute similarity matrix
    delta1,delta2 = deltas
    A = np.exp(-0.5 * (D1**2 / delta1**2 + D2**2 / delta2**2))

    return A

def log_distance(p1,p2):
    output = math.log10(p1/p2)
    return math.fabs(output)

def haversine_distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d
