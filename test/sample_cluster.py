import pytest
import numpy as np
from astropy.table import Table, Column
import limepy

def random_cluster(N, **kwargs):
	"""
	Return a completely random and unphysical stellar cluster.

	Parameters
	----------
		N : int,
			Number of stars to sample
		
	Kwargs
	------
		Keys are column names for the cluster (e.g., m, x, vx, ...)
		Values are str or int:
		'i' : the column is generated from np.arange(N)
		vmax : the column is generated from np.random.randint(0, vmax, size=N)
		'f': the column is generated from np.random.rand(N)

	Return
	------
		tab : astropy Table,
			Stellar cluster with arbitrary columns and units.

	Example
	-------
		random_cluster(100, id='i', m='f', ks=14, x='f')
	"""
	tab = Table(meta={'name': 'cluster'})
	for key, value in kwargs.iteritems():
		if (value == 'i'):
			tab.add_column(Column(np.arange(N), name=key))
		elif (value == 'f'):
			tab.add_column(Column(np.random.rand(N), name=key))
		elif isinstance(value, int):
			tab.add_column(Column(np.random.randint(0, value, size=N),
				name=key))
		else:
			raise KeyError(str(value) + " invalid argument for " + key)
	return tab

def limepy_cluster(N, seed=0, **kwargs):
	"""
	Return a stellar cluster sampled sampled from a King model.

	Parameters
	----------
		N : int,
			Number of stars to sample
		seed : int, 
			Seed of randomization
		kwargs :
			Values passe to the limepy King model. Default : isotropic (g = 1)
			King model with W0 = 5. See limepy for othe keywords. 

	Return
	------
		tab : astropy Table,
			Stellar cluster with m, x, y, z, vx, vy, vz in astrophysical units.
	"""
	if 'phi0' not in kwargs.keys():
		phi0 = 5
	else:
		phi0 = kwargs['phi0']
		del kwargs['phi0']
	if 'g' not in kwargs.keys():
		g = 1
	else:
		g = kwargs['g']
		del kwargs['g']
	k = limepy.limepy(phi0, g, **kwargs)
	c = limepy.sample(k, seed=seed, N=N)
	m = sample_Kroupa_IMF(N)
	tab = Table([m, c.x, c.y, c.z, c.vx, c.vy, c.vz], 
		names=('m', 'x', 'y', 'z', 'vx', 'vy', 'vz'), 
		meta={'name': 'cluster'})
	return tab

def sample_Kroupa_IMF(N, seed=0, mmin=0.08, mmax=100.0):
	"""
	Return star masses as sampled from a Kroupa (2001) IMF.

	Parameters
	----------
	    N : int,
	        Number of stars 
	"""
	alpha1 = 1.3
	alpha2 = 2.3
	c1 = 1. - alpha1
	c2 = 1. - alpha2
	k1 = 2./c1*(0.5**c1 - mmin**c1)
	if (mmin > 0.5):
	    k1 = 0
	    k2 = 1.0 / c2 * (mmax**c2 - mmin**c2)
	else:
	    k2 = k1 + 1./c2*(mmax**c2 - 0.5**c2)
	if (mmax < 0.5):
	    k1 = 2.0 / c1 * (mmax**c1 - mmin**c1)
	    k2 = k1

	xx = np.random.random(N)
	m = np.zeros_like(xx)
	mask1 = xx < k1/k2
	mask2 = xx >= k1/k2
	m[mask1] = (0.5 * c1 * xx[mask1] * k2 + mmin**c1)**(1./c1)
	m[mask2] = (c2 * (xx[mask2] * k2 - k1) + max(0.5, mmin)**c2)**(1./c2)
	return m