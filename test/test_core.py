import pytest
import numpy as np
from astropy.table import Table, Column

import gcpack as gcp
from .sample_cluster import random_cluster, limepy_cluster

class TestCore():
	def test_com(self):
		m = (1., 1., 1.)
		x = (0., 2., 4.)
		y = (0., 0., 0.)

		# intrinsic com
		z = (0., 0., 0.)
		s = gcp.Snapshot(data=[m, x, y, z], names=['m', 'x', 'y', 'z'])
		assert gcp.center_of_mass(s) == [2., 0., 0.]

		# masked com
		i = (0, 1, 2)
		s = gcp.Snapshot(data=[i, m, x, y, z], names=['id', 'm', 'x', 'y', 'z'])
		s.filter(id=(0,1), by_range={"id":False})
		assert gcp.center_of_mass(s, masked=True) == [1., 0., 0.]

		# projected com
		z = (8., 10., -99.)
		s = gcp.Snapshot(data=[m, x, y, z], names=['m', 'x', 'y', 'z'], 
			project=True)
		assert gcp.center_of_mass(s) == [2., 0.]

	def test_local_density(self):
		# produce a test cluster with 10 stars (separated by 1 pc)
		# with equal masses (=1 Msun)
		tab = Table()
		tab['x'] = np.arange(0., 10., 1) # all the stars along x
		tab['y'] = np.zeros_like(tab['x']) * 0.
		tab['z'] = np.zeros_like(tab['x']) * 0.
		tab['m'] = np.ones_like(tab['x']) * 1. # m = 1 for each star

		# not masked
		s = gcp.Snapshot(tab)
		s.add_local_density()
		neighbors_radii = np.array([6., 5., 4., 3., 3., 3., 3., 4., 5., 6.])
		rho = 5. / (4./3 * np.pi * neighbors_radii**3.)
		assert all([s["_rho"][i] == rho[i] for i in range(10)])

		# masked
		s_mskd = gcp.Snapshot(tab)
		s_mskd.filter(x=(0.,8.1))
		s_mskd.add_local_density()
		neighbors_radii = np.array([6., 5., 4., 3., 3., 3., 4., 5., 6.])
		rho = 5. / (4./3 * np.pi * neighbors_radii**3.)
		assert all([s_mskd["_rho"][i] == rho[i] for i in [6, 7, 8]])

	def test_density_center(self):
		# produce a test cluster with 10 stars (separated by 1 pc)
		# with equal masses (=1 Msun)
		tab = Table()
		tab['x'] = np.arange(0., 10., 1) # all the stars along x
		tab['y'] = np.zeros_like(tab['x']) * 0.
		tab['z'] = np.zeros_like(tab['x']) * 0.
		tab['m'] = np.ones_like(tab['x']) * 1. # m = 1 for each star

		s = gcp.Snapshot(tab)
		s.add_local_density()
		rho = s["_rho"]
		dc = gcp.density_center(s)
		assert dc[0] == np.dot(tab['x'], rho) / np.sum(rho)

	def test_lagr_radii(self):
		tab = random_cluster(100, x='f', y='f', z='f')
		s = gcp.Snapshot(tab)
		# what if an incomplete cluster is passed?
		with pytest.raises(ValueError):
			gcp.lagr_rad(s, 50.)

		# produce a realistic GC with rh = 2.5pc
		rh = 2.5
		tab = limepy_cluster(10000, seed=np.random.randint(0,999), M=33070, rh=rh)
		# pass it to an instance of Snapshot
		s = gcp.Snapshot(tab)
		# test if numbers make sense, namely the relative error for the 
		# half-mass radius of a N=10K better be less than 10%
		rh_calc = gcp.lagr_rad(s, 50.)
		assert abs(2. * (rh_calc - rh)/(rh_calc + rh)) < 0.1
		# what if the lagr. percentage is 100 or greater ?
		assert gcp.lagr_rad(s, 100.) >= np.max(s["_r"])
		assert gcp.lagr_rad(s, 105.) == gcp.lagr_rad(s, 100.) 
		s.filter(_r=(0.,gcp.lagr_rad(s, 100.)))
		assert len(s[:]) == 10000

		# what if an almost empty cluster is passed?
		tab = random_cluster(2, m='f', x='f', y='f', z='f')
		s = gcp.Snapshot(tab)
		rh = gcp.lagr_rad(s, 50.) 
		assert np.isnan(rh)

		# what if a small (but not almost empty) cluster is passed?
		tab = random_cluster(3, m='f', x='f', y='f', z='f')
		s = gcp.Snapshot(tab)
		lagrs = gcp.lagr_rad(s, np.arange(100.)) 
		assert not np.all(np.isnan(lagrs))

	def test_density_radius(self):
		# produce a test cluster with 10 stars (separated by 1 pc)
		# with equal masses (=1 Msun)
		tab = Table()
		tab['x'] = np.arange(0., 10., 1) # all the stars along x
		tab['y'] = np.zeros_like(tab['x']) * 0.
		tab['z'] = np.zeros_like(tab['x']) * 0.
		tab['m'] = np.ones_like(tab['x']) * 1. # m = 1 for each star

		s = gcp.Snapshot(tab)
		s.add_local_density()
		rho = s["_rho"]

		# default density radius
		rd = gcp.density_radius(s)
		assert rd == np.dot(tab['x'], rho * rho) / np.sum(rho * rho)

		# mass weighted density radius
		rd = gcp.density_radius(s, mass_weighted=True)
		assert rd == np.dot(tab['x'], rho) / np.sum(rho)

	def test_velocity_dispersion(self):
		# produce a test cluster 
		tab = random_cluster(3, id='i', x='f', y='f', z='f')
		tab['vx'] = [1., 1., 1.]
		tab['vy'] = [1., 2., 2.]
		tab['vz'] = [1., 2., 2.]

		# intrinsic
		s = gcp.Snapshot(tab)
		assert gcp.velocity_dispersion(s)[0] == np.sqrt(np.std([3.,9.,9.]))

		# masked
		s_mskd = gcp.Snapshot(tab)
		s_mskd.filter(id=(0,1), by_range={"id":False})
		assert gcp.velocity_dispersion(s_mskd)[0] == np.sqrt(np.std([3.,9.]))

		# projected
		s.project = True
		assert gcp.velocity_dispersion(s)[0] == np.sqrt(np.std([2.,5.,5.]))

		# los
		assert gcp.velocity_dispersion(s, los=True)[0] == np.sqrt(np.std([1.,4.,4.]))

		# empty cluster
		tab = Table(names=('x','y','z','vx','vy','vz'))
		s = gcp.Snapshot(tab)
		assert np.isnan(gcp.velocity_dispersion(s)[0])
		assert np.isnan(gcp.velocity_dispersion(s)[1])

	def test_density(self):
		# produce a test cluster with 5 stars (separated by 1 pc)
		# with equal masses (=1) and luminosity (=1)
		tab = Table()
		tab['x'] = np.arange(0., 5., 1) # all the stars along x
		tab['y'] = np.zeros_like(tab['x']) * 0.
		tab['z'] = np.zeros_like(tab['x']) * 0.
		tab['m'] = np.ones_like(tab['x']) * 2. # m = 1 for each star
		s = gcp.Snapshot(tab)

		# intrinsic number count
		assert gcp.density(s) == 5. / (4./3. * np.pi * tab['x'][-1]**3.)

		# intrinsic mass 
		assert gcp.density(s, quantity='m')== np.sum(tab['m']) / \
			(4./3. * np.pi * tab['x'][-1]**3.)

		# wrong quantity for density
		with pytest.raises(ValueError):
			tmp = gcp.density(s, quantity='xxxx')

		# masked cluster
		s.filter(_r=(0., 3.1))
		assert gcp.density(s) == 4. / (4./3. * np.pi * tab['x'][-2]**3.)

	def test_mass_function(self):
		# produce a test cluster with 2 star populations
		tab = random_cluster(100, id='i', x='f', y='f', z='f')
		tab["m"] = np.ones(len(tab))
		tab["m"][50:] = 2.

		s = gcp.Snapshot(tab)
		assert np.all(gcp.mass_function(s, bins=2)[0] == np.array([50, 50]))