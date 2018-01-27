import numpy as np
import copy


def vdp(snapshot, radial_bins=None, los=False):
    """ 
    Calculate velocity dispersion profile: sigma = sigma(r)
    Arguments:
        snapshot    gcpack.Snapshot, stellar cluster snapshot

        radial_bins array-like, with the edges of the radial binning. 
                    default binning is defined by 
                    np.linspace(1., 99., num=20)

        los         bool, if True returns the line-of-sight 
                    velocity dispersion profile

    Returns:
        r           array [len(radial_bins)], mean radial position
        sigma       array [len(radial_bins)], velocity dispersion
        sigma_err   array [len(radial_bins)], velocity dispersion error
    """
    snap = copy.deepcopy(snapshot) # make a copy
    if not snap.sort_flag:
        raise ValueError('Cluster must be radially sorted')

    # set radial binning
    if radial_bins is None:
        radii = snap.lagr_r(np.linspace(1., 99., num=20))
        radial_bins = np.array(
            [(radii[i],radii[i+1]) for i in range(len(radii)-1)])
    
    # calculate profile
    sigma, dsigma, r = [], [], []
    for rad_bin in radial_bins:
        snap.select_stars(r=rad_bin)
        radii = snap.get_r(mask=True)
        y, dy = snap.sigma(los=los)
        sigma.append(y)
        dsigma.append(dy)
        r.append(np.average(radii))
    return np.array(r), np.array(sigma), np.array(dsigma)

def vdmp(snapshot, mass_bins=None, los=False):

    """ 
    Calculate velocity dispersion mass profile: sigma = sigma(m)
    Arguments:
        snapshot    gcpack.Snapshot, stellar cluster snapshot

        mass_bins   array-like, with the edges of the mass binning. 
                    If None, use 8 bins equally spaced on a log10 scale.

        los         bool, if True returns the line-of-sight 
                    velocity dispersion mass profile

    Returns:
        m           array [len(mass_bins)], mean mass
        sigma       array [len(mass_bins)], velocity dispersion
        sigma_err   array [len(mass_bins)], velocity dispersion error
    """
    snap = copy.deepcopy(snapshot) # make a copy
    # set mass binning
    if mass_bins is None:
        logm = np.log10(snap.get_selection()['m'])
        logmi = np.linspace(min(logm), max(logm), num=9) 
        mass_bins = np.array(
            [(logmi[i],logmi[i+1]) for i in range(len(logmi)-1)]
            )

    # calculate profile
    sigma, dsigma, m = [], [], []
    for mass_bin in mass_bins:
        snap.select_stars(m=10**mass_bin)
        y, dy = snap.sigma(los=los)
        sigma.append(y)
        dsigma.append(dy)
        m.append(np.average(snap.get_selection()['m']))
    return np.array(m), np.array(sigma), np.array(dsigma)


def dp(snapshot, quantity=None, radial_bins=None):
    
    """ 
    Calculate density profile: n(r), rho(r) or l(r)
    Arguments:
        snapshot    gpack.Snapshot, stellar cluster snapshot

        radial_bins array-like, with the edges of the radial binning. 
                    default binning is defined by 
                    np.linspace(1., 99., num=20)

        quantity    str or None, it is passed to gpack.Snapshot.density.

    Returns:
        r           array [len(radial_bins)], mean radius
        d           array [len(radial_bins)], density
    """

    snap = copy.deepcopy(snapshot) # make a copy 
    
    # set radial binning
    if radial_bins is None:
        radii = snap.lagr_r(np.linspace(1., 99., num=20))
        radial_bins = np.array(
            [(radii[i],radii[i+1]) for i in range(len(radii)-1)])
    
    # calculate profile
    d, r = [], []
    for rad_bin in radial_bins:
        snap.select_stars(r=rad_bin)
        rad, y = snap.density(quantity=quantity)
        d.append(y)
        r.append(rad)
    
    return np.array(r), np.array(d)
