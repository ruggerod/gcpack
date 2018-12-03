import numpy as np
from sklearn.neighbors import KDTree
from scipy.optimize import least_squares
from copy import deepcopy
import warnings

from .lin_regr import lregr

__all__ = [
    "get_local_density", "center_of_mass", "density_center",
    "lagr_rad", "density_radius", "velocity_dispersion", "density",
    "mass_function", "mass_function_slope", "hm_relax_time", "core_radius"
]


def get_local_density(snap, n_neighbors=6, masked=True):
    """
    Calculate local mass densities of each star in a cluster, following
    Casertano & Hut (1985) (see also de Vita et al. 2018). 

    Parameters
    ----------
        snap : Snapshot, 
            Represents the stellar cluster (projected or intrinsic)
        n_neighbors : int, optional
            Number of neighbors for the local density
        masked : bool, optional
            If True, consider only selected stars for calculation 

    Return
    ------
        rho: ndarray with shape (N,), 
            Local densities 
    """
    snap._check_features('m', function_id='Local Density')

    if masked:
        tab = snap[:]
    else:
        tab = snap.original

    if snap.project:
        X = np.array([tab['x'], tab['y']]).T # shape(N, 2)
    else:
        X = np.array([tab['x'], tab['y'], tab['z']]).T # shape(N, 3)

    m = tab['m'] # shape(N, )

    # initialize a KDTree 
    tree = KDTree(X, leaf_size=100)

    # get distances and indeces of the first n_neighbors for each star
    dist_nb, ind_nb = tree.query(X, k=n_neighbors+1)
    
    # sum the masses of all the stars that are neither the central star nor
    # the most distant neighbor.
    Mass = np.sum(m[ind_nb], axis=1) - m - m[ind_nb][:,-1] 
    # calculate area (or volume)
    if snap.project:
        Den = np.pi * dist_nb[:,-1]**2.
    else:
        Den = 4./3. * np.pi * dist_nb[:,-1]**3.

    return Mass / Den


def center_of_mass(snap, masked=True):
    """
    Return the center of mass of a snapshot.

    Parameters
    ----------
    snap : gcpack.Snapshot,
        Snapshot used for center of mass calculation
    masked : bool, optional
        If True, consider only selected stars for the calculation

    Return
    ------
        com : list, 
            Center of mass
    """
    # check against required quantities
    snap._check_features('m', function_id='Center of Mass')

    # get quantities as Table
    if masked:
        tab = snap[:]
    else:
        tab = snap.original

    # get position as one single array
    if snap.project:
        X = np.array([tab['x'], tab['y']]).T
    else:
        X = np.array([tab['x'], tab['y'], tab['z']]).T

    # calculate center of mass
    Mtot = np.sum(tab['m'])
    return [np.dot(X[:,i], tab['m']) / Mtot for i in range(X.shape[1])]


def density_center(snap, masked=True):
    """
    Return density center (Casertano & Hut 85).

    Parameters
    ----------
        snap : gcpack.Snapshot,
            Snapshot used for center of mass calculation
        masked : bool, optional
            If True, consider only selected stars for the calculation

    Return
    ------
        dc : list, 
            Density center
    """
    # get quantities as Table
    if masked:
        tab = snap[:]
    else:
        tab = snap.original

    # check if snapshot has local density and in case, calculate it!
    if "rho" in snap.original.colnames: # rho provided by user at initialization
        rho = tab['rho']
    else:
        try: # rho already calculated by user 
            if masked:
                rho = snap['_rho']
            else:
                rho = snap._data['_rho']
        except: # rho needs to be calculated
            rho = get_local_density(snap, masked=masked)

    # get position as one single array
    if snap.project:
        X = np.array([tab['x'], tab['y']]).T
    else:
        X = np.array([tab['x'], tab['y'], tab['z']]).T

    # calculate density center
    Rhotot = np.sum(rho)
    return [np.dot(X[:,i], rho) / Rhotot for i in range(X.shape[1])]


def lagr_rad(snap, percs, masked=True, light=False):
    """
    Return Lagrangian radii specified by percs.

    Parameters
    ----------
        snap : Snapshot, 
            Represents the stellar cluster 
        percs : numeric, list
            Percentages used to calculate Lagrangian radii
        masked : bool, optional
            If True, consider only masked snapshot for calculation 
        light : bool, optional
            If True, return light-based Lagrangian radii

    Return 
    ------
        lagr_radii : list,
            Lagrangian radii. The i-th element correspond to percs[i].
    """
    # check against required columns
    if light:
        qnt = 'l'
    else:
        qnt = 'm'
    snap._check_features(qnt, function_id='Lagrangian radii')

    # force percs to be a tuple
    try:
        percs = tuple(percs) 
    except:
        percs = (percs, )

    # get radius and mass
    if masked:
        r = snap['_r']
        q = snap[qnt]
    else:
        r = snap._data['_r']
        q = snap._data[qnt]

    # dummy case : empty (or almost) cluster
    if len(r) <= 2: 
        warnings.warn("Too few stars!", RuntimeWarning)
        if len(percs) == 1:
            return np.nan
        else:
            return [np.nan for perc in percs]

    # sort by radius
    so = np.argsort(r)
    r_sort = r[so]
    q_sort = q[so]

    # use bisection method for Lagrangian radii calculations
    Q = np.sum(q_sort)
    lagr_radii = []
    for perc in percs:
        Qperc = Q * perc / 100. # fraction of total mass
        def diff(n): 
            # calculate the difference between the global mass of the 
            # first n elements and Mperc
            return np.sum(q_sort[:n+1]) - Qperc
        a = 0
        b = len(q_sort) - 1
        Nguess = a + (b-a)/2

        # bisection method
        while True:
            if perc >= 100.: # truncation radius contains the max radius
                lagr_radii.append(r_sort[-1]*(1+1e-10))
                break

            dm_guess = diff(Nguess) 
            if dm_guess > 0: # lagr radius inside radius of Nguess-th star
                b = Nguess
                if ((b - a) == 1):
                    lagr_radii.append((r_sort[Nguess] + r_sort[Nguess-1]) / 2.)
                    break
                Nguess = a + (b-a)/2 
            else: # lagr radius outside r[Nguess]
                a = Nguess
                if ((b - a) == 1):
                    lagr_radii.append((r_sort[Nguess] + r_sort[Nguess+1]) / 2.)
                    break
                Nguess = a + (b-a)/2
    if len(lagr_radii) == 1:
        return lagr_radii[0]
    return lagr_radii


def density_radius(snap, masked=True, mass_weighted=False):
    """
    Return density radius.

    Parameters
    ----------
        snap : gcpack.Snapshot, 
            Represents the stellar cluster.
        masked : bool, optional
            If True, consider only masked snapshot for calculation.
        mass_weighted : bool, optional
            If True, calculate density radius as in de Vita et al. 2018.
            Else, calculate as in Casertano & Hut 1985.

    Return
    ------
        rd : float, 
            Density radius
    """
    # define a mask of shape (N, )
    if masked:
        mask = snap._data['_mask']
    else:
        mask = np.ones(len(snap.original), dtype=bool)

    # check if snapshot has local density and in case, calculate it!
    if "rho" in snap.original.colnames: # rho provided by user at initialization
        rho = snap._data['rho'][mask]
    else:
        try: # either rho already calculated by user 
            rho = snap['_rho']
        except: # or rho has to be calculated
            rho = get_local_density(snap, masked=masked)

    # calculate density radius
    if mass_weighted:
        weigths = rho * snap._data['m'][mask]
    else:
        weigths = rho * rho
    num = np.dot(snap._data['_r'][mask], weigths)
    den = np.sum(weigths)
    return num / den


def velocity_dispersion(snap, dim=3, masked=True, isotropy=False):
    """
    Return velocity dispersion. 
    For each dimension i:
    sig_i = mean((v_i - mean(v_i))^2)
    The combined velocity dispersion is:
    sig = sqrt(sum(sig_i^2))
    In case of isotropy, sig_i = const for each i.

    Parameters
    ----------
        snap : gcpack.Snapshot, 
            Represents the stellar cluster.
        masked : bool, optional
            If True, consider only masked snapshot for calculation.
        dim : int, optional
            Velocity dispersion dimension. If isotropy is False,
            1D vel. dispersion is calculated from the velocities along z,
            while 2D vel. dispersion from the velocities along x a y.
        isotropy: bool, optional
            If True, the velocity dispersion is obtained by 
            rescaling the 3D vel. dispersion by sqrt(dim/3).

    Return
    ------
        sig : float,
            Velocity dispersion
        dsig : float,
            Standard error of sig

    Notes
    -----
        If the snapshot is projected the velocity dispersion is calculated only
        with components vx, vy
    """
    # check required quantities
    assert (dim > 0 and dim <= 3), "Wrong dimensions for velocity dispersion!"
    snap._check_features('vz', function_id='Velocity dispersion')
    if dim > 1:
        snap._check_features('vx', 'vy', function_id='Velocity dispersion')

    # get quantities as Table
    if masked:
        tab = snap[:]
    else:
        tab = snap.original

    # define velocity denpending on dimension and isotropy parameters
    if (dim == 1) and (not isotropy):
        V = tab['vz']
    elif (dim == 2) and (not isotropy):
        V = np.array([tab['vx'], tab['vy']]).T
    else:
        V = np.array([tab['vx'], tab['vy'], tab['vz']]).T

    SIG = np.std(V, axis=0)  # velocity dispersion vector
    sig = np.sqrt(np.sum(SIG * SIG))  # quadrature sum
    # rescale 3D velocity dispersion in case of isotropy
    if isotropy:
        sig = sig * np.sqrt(dim / 3.)

    return sig, np.sqrt(sig**2. / (2. * len(tab)))


def density(snap, quantity=None, masked=True):
    """
    Return density (or surface density) of a specified quantity. 

    Parameters
    ----------
        snap : gcpack.Snapshot, 
            Represents the stellar cluster.
        quantity : str, optional
            Snapshot column used for the density calculation (e.g. 'm', 'l').
            If no quantity is provided a number count density is calculated.
        masked : bool, optional
            If True, consider only masked snapshot for calculation.

    Return
    ------
        dens : float,
            Density
    """
    # define a mask of shape (snap.N, )
    if masked:
        mask = snap._data['_mask']
    else:
        mask = np.ones(len(snap.N), dtype=bool)

    # check required quantities
    if quantity is not None:
        snap._check_features(quantity, function_id='Density')
        numerator = np.sum(snap._data[quantity][mask])
    else:
        numerator = len(snap._data[:][mask])

    r = snap._data["_r"][mask]
    rmax = np.max(r)
    rmin = np.min(r)
    if snap.project:
        denominator = np.pi * (rmax**2. - rmin**2.)
    else:
        denominator = 4./3. * np.pi * (rmax**3. - rmin**3.)
    return numerator / denominator


def mass_function(snap, masked=True, **kwargs):
    """
    Return the output of numpy.histogram(m, **kwargs), 
    where m are the stellar masses in the snapshot.

    Parameters
    ----------
        snap : gcpack.Snapshot, 
            Represents the stellar cluster.
        masked : bool, optional
            If True, consider only masked snapshot for calculation.
        kwargs : numpy.histogram properties

    Return
    ------
        hist : array
            The values of the histogram.
        bin_edges : array of dtype float
            Return the bin edges (length(hist)+1).

    """
    # check required quantities
    snap._check_features('m', function_id='Mass Function')

    # get masses
    if masked:
        m = snap['m']
    else:
        m = snap.original['m']

    return np.histogram(m, **kwargs) 


def mass_function_slope(snap, masked=True, nbins=10, plot=False):
    """
    Return slope and error of the Mass Function.

    Parameters
    ----------
        snap : gcpack.Snapshot, 
            Represents the stellar cluster.
        masked : bool, optional
            If True, consider only masked snapshot for calculation.
        nbins : int, optional
            number of bins used for the Mass Function. Bins are chosen in order
            to have approximately the same numbe of stars (see Maiz & Ubeda 2005).
        plot : bool, optional
            Plot the linear fit

    Return
    ------
        alpha : float, 
            Mass Function slope
        error : float,
            Mass Function slope error from linear fit
    """
    # get masses
    if masked:
        m = snap['m']
    else:
        m = snap.original['m']
    N = len(m)

    # define variable mass bins by fixing the number of stars in the bin
    # (see Maiz & Ubeda 2005)
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

    m_sort = np.sort(m) # sorted masses
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


def hm_relax_time(snap, masked=True):
    """
    Return half-mass relaxation time (see eq.3 in Trenti+ 10)

    Parameters
    ----------
        snap : gcpack.Snapshot, 
            Represents the stellar cluster.
        masked : bool, optional
            If True, consider only masked snapshot for calculation.

    Return
    ------
        trh : float, 
            Half-mass relaxation time
    """
    # get number of stars
    if masked:
        N = len(snap[:])
    else:
        N = snap.N

    # calculate half-mass radius
    rh = lagr_rad(snap, 50., masked=masked)

    return 0.138 * N * rh**1.5 / np.log(0.11 * N)


def core_radius(snap, masked=True, light=False):
    """
    Calculates the projected core radius as the radius at which
    the surface brightness ('mu') profile drops to half of its central value.

    Parameters
    ----------
        snap : Snapshot,
            Represents the stellar cluster
        masked : bool, optional
            If True, consider only masked snapshot for calculation.
        light : bool, optional
            If True, returns light-based core radius using the luminosity 'l'
            If False, returns mass-based core radius using the mass 'm'

    Return
    ------
        core radius : float
    """
    s = deepcopy(snap)  # make local copy
    if not s.project:  # make sure to project snapshot
        s.project = True

    if light:
        s._check_features('l', function_id='Core radius')
        s._data['_q'] = s._data['l']
    else:
        s._check_features('m', function_id='Core radius')
        s._data['_q'] = s._data['m']

    # get original mask
    if masked:
        mask = s._data['_mask']
    else:
        s.filter()
        mask = s._data['_mask']

    # calculate surface brightness profile...
    # ... set default radial binning
    radii = lagr_rad(s, np.linspace(1., 100., num=25))
    r5 = radii[1]
    radial_bins = np.array([(radii[i],radii[i+1]) for i in range(len(radii)-1)])
    # ...calculate profile
    dens, rad = [], []
    for rad_bin in radial_bins:
        s.filter(_r=rad_bin)  # radial mask
        s._data['_mask'] = np.logical_and(s._data['_mask'], mask)  # refine mask
        radii = s["_r"]
        rmin, rmax = np.min(radii), np.max(radii)
        area = np.pi * (rmax**2. - rmin**2.)
        dens.append(np.sum(s['_q']) / area)
        rad.append(np.mean(radii))
    r, mu = np.array(rad), np.array(dens)

    def king_res(a, x, y):  # residuals (see eq 14 King '62)
        k, rc, C = a
        return (k * (1/np.sqrt(1+(x/rc)**2.) - C)**2. - y)

    k, Rc, C = least_squares(
        king_res, [mu[0], r5, 0], loss='soft_l1', f_scale=0.1, args=(r, mu)
    )['x']

    return Rc
