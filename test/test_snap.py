import pytest
import gcpack
import numpy as np
from astropy.table import Table, Column
import sys
import limepy

## AUXILIARY FUNCTIONS

def random_cluster(N, **kwargs):
	"""
	Return a Table [N, len(kwargs)].
	As key specify the name of the column feature. As value choose among:
	'i': np.arange(N) is generated
	vmax: np.random.randint(0, vmax, size=N)
	'f': np.random.rand(N) is generated

	Example:
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
	Return a Table [N, M], with M = 7 no. of star features.
	Features are m, x, y, z, vx, vy, vz.
	Kwargs are passed to the limepy model.
	"""
	if 'phi0' not in kwargs.keys():
		phi0 = 5
	else:
		phi0 = kwargs['phi0']
	if 'g' not in kwargs.keys():
		g = 1
	else:
		g = kwargs['g']
	k = limepy.limepy(phi0, g, **kwargs)
	c = limepy.sample(k, seed=seed, N=N)
	tab = Table([c.m, c.x, c.y, c.z, c.vx, c.vy, c.vz], 
		names=('m', 'x', 'y', 'z', 'vx', 'vy', 'vz'), 
		meta={'name': 'cluster'})
	return tab

## TESTING 

class TestSnap():
	def test_default_init(self):
		tab = random_cluster(100, id='i', ks=14, x='f', y='f', z='f',
			vx='f', vy='f', vz='f')
		s = gcpack.Snapshot(tab)
		# 1. is the result of get_selection() the same as the input table?
		assert np.all(s.get_selection() == tab)

		tab = random_cluster(100, m='f')
		# 2. what if an dummy cluster with one column is passed ?
		with pytest.raises(ValueError):
			s = gcpack.Snapshot(tab)

		tab = random_cluster(100, x='f', y='f')
		# 3. what if a not projected cluster is passed with only two coords ?
		with pytest.raises(ValueError):
			s = gcpack.Snapshot(tab)


	def test_selection(self):
		tab = random_cluster(10000, id='i', ks=14, m='f', x='f', y='f', z='f',
			vx='f', vy='f', vz='f')
		# 1. default selection
		s = gcpack.Snapshot(tab, m=(0.2,0.5))
		# are elements in a given range ?
		assert np.all(s.get_selection()['m'] < 0.5)
		assert np.all(s.get_selection()['m'] >= 0.2)

		# 2. overwriting selection
		s.select_stars(m=(0.6,1.0))
		# are element in a range different from before ?
		assert np.all(s.get_selection()['m'] >= 0.6)
		assert np.all(s.get_selection()['m'] < 1.0)

		# 3. inplace selection
		s.select_stars(overwrite=False, id=np.arange(100))
		# are element in the same range as before, but also with the new 
		# selection criterium applied ?
		assert np.all(s.get_selection()['m'] >= 0.6)
		assert np.all(s.get_selection()['m'] < 1.0)
		assert np.all(s.get_selection()['id'] <= 100)

		# 4. wrong selection
		# 4.a wrong key
		with pytest.raises(KeyError):
			s.select_stars(mass=(0.,0.5))
		# 4.b wrong values
		with pytest.raises(ValueError):
			s.select_stars(m=0.5)
		with pytest.raises(ValueError):
			s.select_stars(m='0.5')
		with pytest.raises(ValueError):
			s.select_stars(m=['0.5','0.8'])
		with pytest.raises(ValueError):
			s.select_stars(m=[0.8, 1.0, 1.4])

	def test_sorting(self):
		tab = random_cluster(100, id='i', ks=14, x='f', y='f', z='f',
			vx='f', vy='f', vz='f') # create random cluster
		# evaluate radius from center = (0,0,0) and sort
		r_true = np.sqrt(tab['x']**2. + tab['y']**2. + tab['z']**2.) 
		R_true = np.sqrt(tab['x']**2. + tab['y']**2.) # projected
		r_true_sorted = np.sort(r_true)
		R_true_sorted = np.sort(R_true)
		so_r = np.argsort(r_true) # indeces that would sort r_true
		so_R = np.argsort(R_true) # indeces that would sort R_true

		# create sorted (not projected) snapshot
		s = gcpack.Snapshot(tab, sort=True) 
		# evaluate radius without further sorting
		r_sorted = s.get_r()

		# 1. is r obtained from the sorted selection really sorted?
		assert np.all(r_sorted == r_true_sorted)
		# 2. are the other features sorted consistently?
		assert np.all(s.get_selection() == tab[so_r])

		# create sorted and projected snapshot
		s = gcpack.Snapshot(tab, sort=True, project=True) 
		# get radius
		R_sorted = s.get_r()

		# 3. is R obtained from the sorted selection really sorted?
		assert np.all(R_sorted == R_true_sorted)
		# 4. are the other projected features sorted consistently?
		assert np.all(s.get_selection() == tab[so_R])

	def test_reset_center(self):
		tab = random_cluster(100, id='i', ks=14, x='f', y='f', z='f',
			vx='f', vy='f', vz='f') # create random cluster
		# create a non-projected snapshot
		s = gcpack.Snapshot(tab)
		# 1. is default center a tuple?
		assert type(s.get_center()) == tuple

		# 2. input has len 2 and cluster is not projected
		with pytest.raises(ValueError):
			assert s.reset_center((0., 0.4))

		# create a projected snapshot
		s = gcpack.Snapshot(tab, project=True)
		# 3. input has len 3 and cluster is projected
		with pytest.raises(ValueError):
			assert s.reset_center((0., 0.4, 0.8))
		# 4. input is non sense
		with pytest.raises(ValueError):
			assert s.reset_center('0,0.4')
		# 5. does the center remain a tuple?
		assert type(s.get_center()) == tuple

	def test_com(self):
		tab = random_cluster(100, x='f', y='f', z='f')
		s = gcpack.Snapshot(tab)
		# 1. what if an incomplete cluster is passed?
		with pytest.raises(ValueError):
			s.com()

		tab = random_cluster(100, x='f', y='f', z='f', m='f')
		s = gcpack.Snapshot(tab, project=True)
		# 2. is len(CoM) = 3 despite the projection ?
		assert len(s.com()) == 3
		# 3. if projected, is the third component of CoM a nan ?
		assert np.isnan(s.com()[-1])

	def test_lagr_radii(self):
		tab = random_cluster(100, x='f', y='f', z='f')
		s = gcpack.Snapshot(tab, sort=True)
		# 1. what if an incomplete cluster is passed?
		with pytest.raises(ValueError):
			s.lagr_r(50.)

		# produce a realistic GC with rh = 2.5pc
		rh = 2.5
		tab = limepy_cluster(1000, seed=0, M=33070, rh=rh)
		# pass it to an instance of Snapshot
		s = gcpack.Snapshot(tab, sort=True)
		# 2. test if numbers make sense, namely the relative error for the 
		# half-mass radius of a N=1K better be less than 10%
		assert abs((s.lagr_r(50.) - rh)/rh) < 0.1

	def test_local_density(self):
		# produce a dummy cluster with 10 stars (separated by 1 pc)
		# with equal masses (=1 Msun)
		tab = Table()
		tab['x'] = np.arange(0., 10., 1) # all the stars along x
		tab['y'] = np.zeros_like(tab['x']) * 0.
		tab['z'] = np.zeros_like(tab['x']) * 0.
		tab['m'] = np.ones_like(tab['x']) * 1. # m = 1 for each star

		# check density of central star
		s = gcpack.Snapshot(tab)
		rho = s.get_local_density()
		assert(rho[0] == 5. / (4./3 * np.pi * tab['x'][6]**3.))

		# check masked density
		# select the 7 stars with x < 7.
		s.select_stars(x=(0.,7.))
		assert(len(s.get_local_density(mask=True)) == 7)


		# add rho to the table
		tab['rho'] = rho
		# create new cluster
		s = gcpack.Snapshot(tab)
		# check if, when present, the local density is returned without 
		# further calculations
		assert np.all(s.get_local_density() == rho)









