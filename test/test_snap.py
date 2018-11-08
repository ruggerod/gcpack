import pytest
import numpy as np
from astropy.table import Table

import gcpack as gcp
from .sample_cluster import random_cluster


class TestSnap():
    def test_default_init(self):
        # 1. right construction 
        tab = random_cluster(100, id='i', kw=14, x='f', y='f', z='f',
            vx='f', vy='f', vz='f')
        s = gcp.Snapshot(tab)
        assert (len(s.original) == len(tab))

        # 2. wrong construction
        with pytest.raises(ValueError):
            s = gcp.Snapshot([np.arange(100)], names=['m'])

    def test_randomization(self):
        # Test rotation (only coordinates)
        tab = random_cluster(1000, x='f', y='f', z='f')
        # original cluster
        s1 = gcp.Snapshot(tab)
        # randomized cluster
        s2 = gcp.Snapshot(tab, seed=999)
        # x, y and z coordinates must all differ
        assert np.all(s1['x'] != s2['x'])
        assert np.all(s1['y'] != s2['y'])
        assert np.all(s1['z'] != s2['z'])
        # but not the radii (allowing a 1e-15 precision error)
        assert np.all(np.abs(s1['_r']-s2['_r']) < np.zeros(len(tab)) + 1e-15)

        # Test rotation (also velocities)
        tab = random_cluster(1000, x='f', y='f', z='f', vx='f', vy='f', vz='f')
        # original cluster
        s1 = gcp.Snapshot(tab)
        # randomized cluster
        s2 = gcp.Snapshot(tab, seed=9)
        # x, y and z coordinates and velocities must all differ
        assert np.all(s1['x'] != s2['x'])
        assert np.all(s1['y'] != s2['y'])
        assert np.all(s1['z'] != s2['z'])
        assert np.all(s1['vx'] != s2['vx'])
        assert np.all(s1['vy'] != s2['vy'])
        assert np.all(s1['vz'] != s2['vz'])
        # but not the radii (allowing a 1e-15 precision error)
        assert np.all(np.abs(s1['_r']-s2['_r']) < np.zeros(len(tab)) + 1e-15)
        # and not the velocities squared
        v1 = np.sqrt(s1['vx']**2. + s1['vy']**2. + s1['vz']**2.)
        v2 = np.sqrt(s2['vx']**2. + s2['vy']**2. + s2['vz']**2.)
        assert np.all(np.abs(v1-v2) < np.zeros(len(tab)) + 1e-15)

    def test_filter(self):
        tab = random_cluster(10000, id='i', kw=14, m='f', x='f', y='f', z='f',
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
        s.filter(m=(0.6,1.0), id=np.arange(50), by_range={'m':True, 'id':False})
        # are element in the same range as before, but also with the new 
        # selection criterium applied ?
        assert np.all(s['m'] < 1.0)
        assert np.all(s['m'] >= 0.6)
        assert np.all(s['id'] <= 50)

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