import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter 

# some plot parameters
plt.rcParams['figure.figsize'] = (10,8) 
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.major.size'] = 10

def plot_vel_disp_prof(r, sig, dsig, rh=None, units=None, savepdf=None):
    """
    Plot velocity dispersion profile

    Parameters
    ----------
        r : numpy array, 
            Radii
        sig : numpy array,
            Velocity dispersion
        dsig : numpy array,
            Velocity dispersion error
        rh : float or None, optional,
            Half-mass radius. If provided the plot is normalized on both axes.
        units : dict {'rstar' : float, 'vstar' : float},
            Astrophysical units used to scale the quantities.
        savepdf : str or None
            Name of the pdf in which to save the figure
    """
    if units is not None:
        rscale = units['rstar']
        vscale = units['vstar']
        xlab = r"$r$ [pc]"
        ylab = r"$\sigma$ [km/s]"
    elif rh is not None:
        rscale = 1. / rh
        vscale = 1. / sig[0]
        xlab = r"$r/r_\mathrm{h}$"
        ylab = r"$\sigma/\sigma(0)$"

    x = r * rscale
    y = sig * vscale
    dy = dsig * vscale
    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=dy, fmt='o')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=0))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=0))
    ax.yaxis.set_minor_formatter(ScalarFormatter(useOffset=0))
    if savepdf is not None:
        plt.savefig(savepdf)
    plt.show()


def plot_vel_disp_mass_prof(m, sig, dsig, units, savepdf=None):
    """
    Plot velocity dispersion mass profile

    Parameters
    ----------
        m : numpy array, 
            Masses
        sig : numpy array,
            Velocity dispersion
        dsig : numpy array,
            Velocity dispersion error
        units : dict {'mstar' : float},
            Astrophysical units used to scale the mass.
        savepdf : str or None
            Name of the pdf in which to save the figure

    Note
    ----
        Warning : the plot is very sensitive to some encoded parameters
    """
    x = m * units['mstar']
    y = sig / sig[0]
    dy = dsig / sig[0]
    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=dy, fmt='o')
    ax.set_xlabel(r"$m$ [M$_\odot$]")
    ax.set_ylabel(r"$\sigma/\sigma(0)$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(np.arange(min(x), max(x), 10.))
    ax.set_xticks(np.arange(min(x), max(x), 1.), minor=True)
    ax.set_xticklabels([], minor=True)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=0))
    ax.set_yticks(np.arange(0.7,1.02,0.05))
    ax.set_yticks(np.arange(0.7,1.01,0.025), minor=True)
    ax.set_yticklabels([], minor=True)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=0))
    if savepdf is not None:
        plt.savefig(savepdf)
    plt.show()

def plot_count_dens_prof(r, n, rh=None, units=None, savepdf=None):
    """
    Plot stellar count density profile

    Parameters
    ----------
        r : numpy array, 
            Radii
        n : numpy array,
            Stellar count density
        rh : float or None, optional,
            Half-mass radius. If provided the plot is normalized on both axes.
        units : dict {'rstar' : float},
            Astrophysical units used to scale to scale the horizontal axis.
        savepdf : str or None
            Name of the pdf in which to save the figure
    """
    if units is not None:
        rscale = units['rstar']
        nscale = units['rstar']**(-3.)
        xlab = r"$r$ [pc]"
        ylab = r"star count [pc$^{-3}$]"
    elif rh is not None:
        rscale = 1. / rh
        nscale = 1. / q[0]
        xlab = r"$r/r_\mathrm{h}$"
        ylab = r"star count [normalized]"

    x = r * rscale
    y = n * nscale
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=0))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=0))
    if savepdf is not None:
        plt.savefig(savepdf)
    plt.show()