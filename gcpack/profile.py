import numpy as np
from copy import deepcopy
from .snapshot import Snapshot
from .core import *

__all__ = [
	"velocity_dispersion_profile", "velocity_dispersion_mass_profile",
    "density_profile"
]

def velocity_dispersion_profile(snap, radial_bins=None, los=False):
    """ 
    Calculate velocity dispersion profile: sigma = sigma(r)

    Parameters
    ----------
        snap : gcpack.Snapshot, 
            Represents the stellar cluster (projected or intrinsic)
        radial_bins : numpy array, list or None, optional
            Edges of the radial binning. Default binning is defined by 
            the lagrangian radii defined by np.linspace(1., 100., num=20)
            percentages.
        los : bool, 
            If True returns the line-of-sight velocity dispersion profile

    Return
    ------
        r : np.array [len(radial_bins)], 
            Average radial position
        sigma : np.array [len(radial_bins)], 
            Velocity dispersion
        sigma_err : np.array [len(radial_bins)], 
            Velocity dispersion error
    """
    snap = deepcopy(snap) # make a copy
    snap.sort('_r')

    # set default radial binning
    if radial_bins is None:
        radii = lagr_rad(snap, np.linspace(1., 100., num=20))
        radial_bins = np.array(
            [(radii[i],radii[i+1]) for i in range(len(radii)-1)])
    
    # calculate profile
    sigma, dsigma, r = [], [], []
    for rad_bin in radial_bins:
        snap.filter(_r=rad_bin)
        radii = snap['_r']
        y, dy = velocity_dispersion(snap, los=los)
        sigma.append(y)
        dsigma.append(dy)
        r.append(np.mean(radii))
    return np.array(r), np.array(sigma), np.array(dsigma)

def velocity_dispersion_mass_profile(snap, mass_bins=None, los=False):
    """ 
    Calculate velocity dispersion mass profile: sigma = sigma(m)

    Parameters
    ----------
        snap : gcpack.Snapshot, 
            Represents the stellar cluster (projected or intrinsic)
        mass_bins : numpy.array, list or None, optional
            Edges of the mass binning. 
            Default is using 8 bins equally spaced on a log10 scale.
        los : bool, 
            If True returns the line-of-sight velocity dispersion profile

    Return
    ------
        m : np.array [len(radial_bins)], 
            Average mass
        sigma : np.array [len(radial_bins)], 
            Velocity dispersion
        sigma_err : np.array [len(radial_bins)], 
            Velocity dispersion error
    """
    snap = deepcopy(snap) # make a copy
    snap.sort('m')

    # set mass binning
    if mass_bins is None:
        logm = np.log10(snap['m'])
        logmi = np.linspace(min(logm), max(logm), num=9) 
        mass_bins = np.array(
            [(logmi[i],logmi[i+1]) for i in range(len(logmi)-1)])

    # calculate profile
    sigma, dsigma, m = [], [], []
    for mass_bin in mass_bins:
        snap.filter(m=10**mass_bin)
        y, dy = velocity_dispersion(snap, los=los)
        sigma.append(y)
        dsigma.append(dy)
        m.append(np.mean(snap['m']))
    return np.array(m), np.array(sigma), np.array(dsigma)


def density_profile(snap, quantity=None, radial_bins=None):
    """ 
    Calculate density (or surface density) profile: dens = dens(r)

    Parameters
    ----------
        snap : gcpack.Snapshot, 
            Represents the stellar cluster (projected or intrinsic)
        quantity : str or None, optional
            Snapshot column used for the density calculation (e.g. 'm', 'l').
            If no quantity is provided a number count density is calculated.
        radial_bins : numpy array, list or None, optional
            Edges of the radial binning. Default binning is defined by 
            the 20 lagrangian radii defined by np.linspace(1., 100., num=20)
            percentages.

    Returns
    -------
        rad : np.array [len(radial_bins)], 
            Average radial position
        dens : np.array [len(radial_bins)], 
            Density (or surface density)
    """
    # check quantity
    if quantity is not None:
        snap._check_features(quantity, function_id='Density profile')

    snap = deepcopy(snap) # make a copy
    snap.sort('_r')

    # set default radial binning
    if radial_bins is None:
        radii = lagr_rad(snap, np.linspace(1., 100., num=20))
        radial_bins = np.array(
            [(radii[i],radii[i+1]) for i in range(len(radii)-1)])

    # calculate profile
    dens, rad = [], []
    for rad_bin in radial_bins:
        snap.filter(_r=rad_bin)
        radii = snap["_r"]
        d = density(snap, quantity=quantity)
        dens.append(d)
        rad.append(np.mean(radii))
    return np.array(rad), np.array(dens)