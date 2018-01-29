# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from astropy.table import Table
import collections

class Snapshot:
    """ Snapshot

    Class that represents a snapshot of N stars with M features.

    Parameters
        snap:       Table of size [N, M]. 
                    Each row is a star; each column a star feature 
                    (see columns' names below)

        project:    bool, 
                    specifies if the snapshot has to be analysed in
                    projection or not; default is False.
                    The convention is that the line of sight is the z-axis.

        sort:       bool, 
                    specifies if the snapshot has to be sorted by the distance 
                    to the center; default is False.

        center:     tuple,
                    specifies the coordinates (x, y, z) of the center of the 
                    system; default is (0, 0, 0)

        **kwargs:   keywords used by select_stars at initialization
    
    Attributes:
        N:             int, total number of stars
        project_flag:  bool, if True the snapshot is projected.
        sort_flag:     bool, if True the snapshot is radially sorted.

    Possible snap columns keywords (in brackets is optional):
        x,y,(z):            position (separate columns)
        (vx), (vy), (vz):   velocity (separate columns)
        (m):                mass
        (id):               simulation id
        (ks):               stellar type
        (l):                luminosity
        (rho):              local density
        (mu):               local surface brightness
        ...
        
    Methods:
        select_stars(overwrite=True, **kwargs)
        get_selection()
        reset_center()
        get_center()
        get_r(mask=False)
        com()
        larg_r()
        sigma(los=False)
        density(quantity=None)
        MF(nbins='auto')
        trh()
        get_local_density(neighbors=6, mask=False)

        # DC()
        # rd()
        # info()
        # alpha(nbins=10, plot=False)
        
    ----------------------------------------------------------------
    
    Private attributes:
        _center:    (x, y, z), tuple that represents the center
        _snap:      Table [N, M+...], each line is a star;
                    each column a star feature. Additional features
                    can be added by specific methods

    """

    def __init__(self, snap, project=False, sort=False, 
        center=(0,0,0), **kwargs):

        self._snap = snap.copy()
        self.project_flag = project
        # check user's features
        self._check_features('Initialization', 'x', 'y')
        if not self.project_flag:
            self._check_features('Initialization', 'z')
        self.sort_flag = sort
        self._center = center

        self.N = len(snap)
        self._snap['mask'] = np.ones(self.N, dtype='bool')
        self.select_stars(**kwargs)
        if self.sort_flag:
            self._rad_sort()

    def select_stars(self, overwrite=True, **kwargs):
        """
        Mask the Snapshot by selecting stars according to different
        criteria. All the criteria specified are combined in a logical 
        union mask.
        If overwrite is True, the masking is applied to the original snapshot. 
        Otherwise, it is applied to the existent selection.

        Masking examples:
            ks:     array-like (1,2,...,14), stellar type
            m:      array-like (mmin, mmax), stellar mass range
            r:      array-like (rmin, rmax), radial range
            id:     array-like (id1, id2, ...), specific ids of stars
        """
        # check for invalid keys
        for key, value in kwargs.iteritems():
            if key not in self._snap.colnames:
                raise KeyError(key + " is an invalid argument")

        # reset selection
        if overwrite: # reset masking
            self._snap['mask'] = np.ones(self.N, dtype='bool')

        self._snap['mask'] = self._mask(**kwargs)

    def get_selection(self):
        """
        Return Table of length = N_reduced, where N_reduced <= N, is the number 
        of stars selected by the user after masking the original snapshot.
        """
        selection = self._snap[:][self._snap['mask']]
        # if present, remove columns that have been added by this class
        selection.remove_column('mask')
        try:
            selection.remove_column('r')
        except:
            pass
        return selection

    def get_center(self):
        """
        Return center coordinates
        """
        return self._center

    def get_r(self, mask=False):
        """
        Return distance to the center of all the stars.
        If mask is True, return only the distance of the selected stars.
        """
        if self.sort_flag:
            if mask:
                radius = self._snap[:][self._snap['mask']]['r']
            else:
                radius = self._snap['r']
        else:
            if self.project_flag:
                radius = np.sqrt(
                    (snap['x']-self._center[0])**2.
                    + (snap['y']-self._center[1])**2.
                    )
            else: 
                radius = np.sqrt(
                    (snap['x']-self._center[0])**2.
                    + (snap['y']-self._center[1])**2.
                    + (snap['z']-self._center[2])**2.
                    )
        return radius

    def reset_center(self, center, sort=False):
        """
        Change the coordinates of the cluster's center and optionally sort.
        """
        # check the new center
        try:
            if self.project_flag:
                if (len(center) != 2):
                    raise ValueError("Provide a center in the form (x, y)")
                else:
                    self._center = (center[0], center[1], np.nan)

            else:
                if (len(center) != 3):
                    raise ValueError("Provide a center in the form (x, y, z)")
                self._center = tuple(center) 
        except:
            raise ValueError(str(center) + " invalid argument!") 
        if sort:
            self._rad_sort()

    def com(self):
        """
        Return center of mass.
        """
        self._check_features('Center of Mass', 'm')

        t = self.get_selection()
        if self.project_flag:
            com = tuple(np.array([
               np.dot(t['x'],t['m']),
               np.dot(t['y'],t['m'])
               ]) / np.sum(t['m'])) + (np.nan,)
        else:
            com = tuple(np.array([
               np.dot(t['x'],t['m']),
               np.dot(t['y'],t['m']),
               np.dot(t['z'],t['m'])
               ]) / np.sum(t['m']))
        return com

    def lagr_r(self, percs):
        """
        Return an array where the i-th element is the radius that contains
        a fraction percs[i] of the global mass.
        """
        self._check_features('Lagrangian radii', 'r', 'm')

        snap = self.get_selection()
        snap['r'] = self.get_r(mask=True)
        #snap = self._snap[:][self._snap['mask']]
        try:
            percs = tuple(percs) 
        except:
            percs = (percs, )
        M = np.sum(snap['m'])
        lagr_radii = []
        for perc in percs:
            Mperc = M * perc / 100. # fraction of total mass
            def diff(n): 
                # calculate the difference between the global mass of the 
                # first n elements and Mperc
                return np.sum(snap['m'][:n+1]) - Mperc
            a = 0
            b = len(snap) - 1
            Nguess = a + (b-a)/2
    
            while True: # bisection method
                dm_guess = diff(Nguess) 
                if dm_guess > 0: # lagr radius inside radius of Nguess-th star
                    b = Nguess
                    if ((b - a) == 1):
                        lagr_radii.append((snap['r'][Nguess] +\
                             snap['r'][Nguess-1]) / 2.)
                        break
                    Nguess = a + (b-a)/2 
                else: # lagr radius outside r[Nguess]
                    a = Nguess
                    if ((b - a) == 1):
                        lagr_radii.append((snap['r'][Nguess] +\
                             snap['r'][Nguess+1]) / 2.)
                        break
                    Nguess = a + (b-a)/2
        if len(lagr_radii) == 1:
            return lagr_radii[0]
        return lagr_radii

    def sigma(self, los=False):
        """
        Return [s,ds] with s velocity dispersion (2D if projected, 
        along the line-of-sight if los=True) and ds standard error of s.
        """
        snap = self.get_selection()
        if los:
            self._check_features('Velocity dispersion', 'vz')
            sig = np.sqrt(np.std(snap['vz'])**2.)
        elif self.project_flag:
            self._check_features('Velocity dispersion', 'vx', 'vy')
            sig = np.sqrt(np.std(snap['vx'])**2. +
                np.std(snap['vy'])**2.)
        else:
            self._check_features('Velocity dispersion', 'vx', 'vy', 'vz')
            sig = np.sqrt(np.std(snap['vx'])**2. +
                np.std(snap['vy'])**2. + np.std(snap['vz'])**2.)

        return sig, np.sqrt(sig**2./(2.*len(snap)))

    def density(self, quantity=None):
        """
        Return density of a specified quantity. Default is number 
        density.
        """
        snap = self.get_selection()
        snap['r'] = self.get_r(mask=True)
        if quantity is None:
            numerator = len(snap) 
        else:
            self._check_features('Density', quantity)
            numerator = np.sum(snap[quantity])

        rad = self.get_r(mask=True)
        if self.project_flag:
            denominator = np.pi * (np.max(rad)**2. 
                - np.min(rad)**2.)
        else:
            denominator = 4./3. * np.pi * (np.max(rad)**3. 
                - np.min(rad)**3.)
        return np.average(rad), numerator / denominator 

    def MF(self, nbins='auto'):
        """
        Return the output of numpy.histogram(m, bins=nbins), 
        where m is the mass of the selected stars.
        """
        snap = self.get_selection()
        self._check_features('Mass function', 'm')
        return np.histogram(snap['m'], bins=nbins) 

    def trh(self):
        """
        Return half-mass relaxation time (see eq.3 in Trenti+ 10)
        """
        N = len(self.get_selection())
        rh = self.larg_r(50.)
        return 0.138 * N * rh**1.5 / np.log(0.11 * N)

    def alpha(self, nbins=10, plot=False):
        """
        Return [alpha, dalpha], with alpha and dalpha respectively 
        slope and error of the Mass Function with nbins mass bins with 
        approximately the same numbe of stars.
        Optionally plots the linear fit.
        """
        # get sorted cluster with radius
        snap = get_selection()
        N = len(snap)

        # define variable mass bins by fixing the number of stars in the bin
        # (see Maiz&Ubeda05)
        stars_per_bin = N // nbins # minimum number of stars per bin
        exceeding_stars = N % nbins # intial number of stars in excess
        nbin = [] # list with len nbins:
                  # each element gives the number of stars in a mass bin
        for n in range(nbins): # for each mass bin...
            if exceeding_stars > 0:
                nbin.append(stars_per_bin + 1)
                exceeding_stars = exceeding_stars - 1
            else:
                nbin.append(stars_per_bin) # leave the fixed number of stars
        np.random.shuffle(nbin) # shuffle to avoid systematics

        m_sort = np.sort(snap['m']) # sorted masses
        xlist, ylist, yerrlist = [], [], []
        for i in range(nbins):
            # get masses in the bin
            ms = m_sort[int(np.sum(nbin[:i])):int(np.sum(nbin[:i+1]))]
            Ni = len(ms) # stars in bin
            dmi = ms[-1] - ms[0] # bin width
            yi = np.log10(Ni/dmi) # value of mass function
            xi = np.log10(np.mean(ms)) # average mass in bin
            wi = Ni * N / (N-Ni) / (np.log10(np.e))**2. # weights
            si = 1. / np.sqrt(wi) # error of log10(Ni)
            yi_err = yi * np.log(np.e) * si 
            ylist.append(yi)
            xlist.append(xi)
            yerrlist.append(yi_err)
        x, y, yerr  = np.array(xlist), np.array(ylist), np.array(yerrlist)

        theta, theta_err = lregr(x, y, yerr, 
            verbose=plot, plot=plot, labels=('log(mass)','log(dN/dm)'))
        return theta[1], theta_err[1]

    def get_local_density(self, neighbors=6, mask=False):
        """
        If present, return the local densities for the selected stars.
        Otherwise, calculate (and then return) the local densities following
        Casertano & Hut '85.

        Arguments:
        neighbors   int, 
                    number of closest stars used to evaluate the local density

        mask        bool,
                    if False the calculation is performed on the entire 
                    cluster, and not only on the subgroup of selected stars.
        """
        try:
            if mask:
                rho = self._snap[:][self._snap['mask']]['rho']
            else:
                rho = self._snap['rho']
            return rho
        except:
            self._check_features('Local density', 'm')

            if mask:
                snap = self.get_selection()
            else:
                snap = self._snap
            m, x, y, z = snap['m'], snap['x'], snap['y'], snap['z']

            rho = np.ones(self.N) * np.nan
            for i in range (len(snap)): #for each star 
                # partially sort the mass array according 
                # to the distances r to the i-th star...

                # find distances to the i-th star
                r = np.sqrt((x - x[i])**2. + (y - y[i])**2. + (z - z[i])**2.)
                # find indeces that would sort the first neighbors of r 
                so = np.argpartition(r, neighbors)
                rs = r[so][:neighbors+1] # distances of the close neighbors
                ms = m[so][:neighbors+1] # masses of the close neighbors

                # finally, the local density
                # note: the total mass at the numerator is calculated excluding 
                # the masses of the i-th star and its neighbors-th closest star.
                rho[i] = (np.sum(ms[:-1]) - m[i]) / \
                    (4./3 * np.pi * rs[neighbors]**3)
        
            self._snap['rho'] = np.array(rho)
            if mask:
                return self._snap[:][self._snap['mask']]['rho']
            else:
                return self._snap['rho']

    def _rad_sort(self):
        """
        Radially sort the cluster with respect to the center
        """
        if self.project_flag:
            radius = np.sqrt(
                (self._snap['x']-self._center[0])**2.
                + (self._snap['y']-self._center[1])**2.
                )
        else: 
            radius = np.sqrt(
                (self._snap['x']-self._center[0])**2.
                + (self._snap['y']-self._center[1])**2.
                + (self._snap['z']-self._center[2])**2.
                )
        self._snap['r'] = radius
        self._snap.sort(keys='r') # sort inplace 

    def _mask(self, **kwargs):
        """
        Return a single bool array, which is the logical union of
        all the selection criteria
        """
        # list of masking criteria: first element is the existent mask
        mask = [self._snap['mask'], ]

        for key, value in kwargs.iteritems():
            # for each masking criterium

            if (np.issubdtype(self._snap[key].dtype, int)): 
                # if the mask is applied on integer numbers ...
                # select only elements in the list provided with value
                mask.append(np.isin(self._snap[key], value)) 
            elif (np.issubdtype(self._snap[key].dtype, float)): 
                # else, if the mask is applied on floats
                # select only elements in the given range
                try:
                    if (len(value) == 2):
                        condition = np.logical_and(
                            self._snap[key] >= value[0],
                            self._snap[key] < value[1]
                            )
                    else: # raise a sure exception
                        tmp = value[len(value)] 
                except:
                    raise ValueError(
                        str(value) + " is an invalid value for masking " \
                        + key)
                mask.append(condition)
        if len(mask) == 1:
            return mask[0]
        if len(mask) > 1:
            return np.logical_and.reduce(mask)

    def _check_features(self, function_name='', *args):
        """
        Raise a ValueError if one of the features specified by args are 
        not column names of the cluster
        """
        missing_features = []
        for feature in args:
            if feature not in self._snap.colnames:
                missing_features.append(feature)
        if len(missing_features) > 0:
            raise ValueError(function_name + ': provide ' 
                + str(missing_features))

    # def DC(self):
    #     """
    #         Returns [x,y,z] or [x,y] density center.
    #     """
    #     t = self.selection
    #     try: # in case the user have a local density column
    #         if self._project:
    #             dc = np.array([
    #                 np.dot(t['x'],t['rho']),
    #                 np.dot(t['y'],t['rho'])
    #                 ]) / np.sum(t['rho'])
    #         else:
    #             dc = np.array([
    #                 np.dot(t['x'],t['rho']),
    #                 np.dot(t['y'],t['rho']),
    #                 np.dot(t['z'],t['rho'])
    #                 ]) / np.sum(t['rho'])
    #     except:
    #         raise ValueError("Error: provide local density!")
    #     return dc

    # def info(self):
    #     """
    #         Print basic information of the snapshot
    #     """
    #     output = collections.OrderedDict()
    #     output['N stars'] = len(self._snap)
    #     if self.sort_flag:
    #         output['Max radius'] = np.max(self._snap['r'])
    #     output['M tot'] = np.sum(self.selection['m'])
    #     output['5   per cent Lagr. rad.'] = self.Lagr_radii()[1][1]
    #     output['10  per cent Lagr. rad.'] = self.Lagr_radii()[2][1]
    #     output['half-mass rad.'] = self.rh()
    #     output['95  per cent Lagr. rad.'] = self.Lagr_radii()[-2][1]
    #     output['100 per cent Lagr. rad.'] = self.Lagr_radii()[-1][1]
    #     if 'rho' in self._snap.columns:
    #         output['density rad.'] = self.rd()
    #     output['half-mass relax. time'] = self.trh()

    #     s = ''
    #     for key in output.keys():
    #         s += ('{:30}'.format(key) + str(output[key]) + '\n')
    #     return s

    # def rd(self):
    #     """
    #          Returns density radius (mass-weighted) (see de Vita+ 18)
    #     """
    #     t = self.selection
    #     dc = self.DC()
    #     if self._project:
    #         radius = np.sqrt(
    #             (t['x']-dc[0])**2.
    #             + (t['y']-dc[1])**2.
    #             )
    #     else:
    #         radius = np.sqrt(
    #             (t['x']-dc[0])**2.
    #             + (t['y']-dc[1])**2.
    #             + (t['z']-dc[2])**2.
    #             )
    #     num = np.sum(radius * t['rho'] * t['m'])
    #     den = np.sum(t['rho'] * t['m'])
    #     return num / den
    # """

