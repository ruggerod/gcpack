import pytest
import numpy as np
from astropy.table import Table

import gcpack as gcp
from .sample_cluster import random_cluster


class TestSnap():
	def test_default_init(self):
		# 1. right construction 
		tab = random_cluster(100, id='i', ks=14, x='f', y='f', z='f',
			vx='f', vy='f', vz='f')
		s = gcp.Snapshot(tab)
		assert (len(s.original) == len(tab))

		# 2. wrong construction
		with pytest.raises(ValueError):
			s = gcp.Snapshot([np.arange(100)], names=['m'])

	def test_filter(self):
		tab = random_cluster(10000, id='i', ks=14, m='f', x='f', y='f', z='f',
			vx='f', vy='f', vz='f')
		# 1. default selection
		s = gcp.Snapshot(tab)
		s.filter(m=(0.2,0.5))
		# are elements in a given range ?
		assert np.all(s['m'] < 0.5)
		assert np.all(s['m'] >= 0.2)

		# 2. overwriting selection
		s.filter(m=(0.6,1.0))
		# are element in a range different from before ?
		assert np.all(s['m'] < 1.0)
		assert np.all(s['m'] >= 0.6)

		# 3. inplace selection
		s.filter(overwrite=False, by_range=False, id=np.arange(100))
		# are element in the same range as before, but also with the new 
		# selection criterium applied ?
		assert np.all(s['m'] >= 0.6)
		assert np.all(s['m'] < 1.0)
		assert np.all(s['id'] <= 100)

		# 4. wrong selection
		# 4.a wrong key
		with pytest.raises(KeyError):
			s.filter(mass=(0.,0.5))
		# 4.b wrong values
		with pytest.raises(ValueError):
			s.filter(m=0.5)
		with pytest.raises(ValueError):
			s.filter(m='0.5')
		with pytest.raises(ValueError):
			s.filter(m=['0.5','0.8'])
		with pytest.raises(ValueError):
			s.filter(m=[0.8, 1.0, 1.4])

	def test_sort(self):
		tab = random_cluster(50, id='i', m='f', x='f', y='f', z='f')
		m_sorted = np.sort(tab['m'])

		s = gcp.Snapshot(tab)
		s.sort('m')
		assert np.all(s['m'] == m_sorted)