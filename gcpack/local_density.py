import numpy as np
from sklearn.neighbors import KDTree

def get_local_densities(snap, n_neighbors=6):
    """
    Calculate the local mass densities of each star in a cluster, following
    Casertano & Hut (1985). 

    Arguments:
        snap    astropy.Table, minimal shape is (N, 4) with N total number of
                stars with columns x, y, z, m
        n_neighbors int, number of neighbors for the local density

    Return:
        rho     np.array, shape(N, ) local densities
    """
    try:
        X = np.array([snap['x'], snap['y'], snap['z']]).T # shape(N, 3)
        m = snap['m'] # shape(N, )
    except:
        raise ValueError("Snapshot must have m, x, y, z columns")

    # initialize a KDTree 
    tree = KDTree(X, leaf_size=100)

    # get distances and indeces of the first n_neighbors for each star
    dist_nb, ind_nb = tree.query(X, k=n_neighbors+1)
    
    # sum the masses of all the stars that are neither the central star nor
    # the most distant neighbor.
    Mass = np.sum(m[ind_nb], axis=1) - m - m[ind_nb][:,-1] 
    Vol = 4./3. * np.pi * dist_nb[:,-1]**3.

    return Mass / Vol