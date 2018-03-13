import numpy as np
from astropy.table import Table
from copy import deepcopy
import gcpack as gcp

class Snapshot(object):
    """ 
    Creates a snapshot of N stars with M features by means of an astropy Table.

    Parameters
    ----------
        data : numpy ndarray, dict, list, Table, or table-like object, optional
            Data to initialize table (see options below). 
        names : list, optional
            Specify column names (see options below).   
        dtype : list, optional
            Specify column data types.     
        project : bool, optional
            Project cluster along z axis 
        center : list, optional
            Stellar system center

    User's columns
    --------------
        x, y, z : required, position as separate columns
        vx, vy, vz : optional, velocity as separate columns
        m : optional, mass
        id : optional, star id
        kw : optional, stellar type
        l : optional, luminosity
        rho : optional, local density
        mu : optional, local surface brightness

    Attributes
    ----------
        N : int, 
            Total number of stars in the cluster
        original : Table, 
            Snapshot provided at initialization
        rad_sorted : Table, 
            Snapshot provided at initialization, but radially sorted
        _data : Table,
            Represents the Snapshot including extra derived columns
        _data['_XXX'] : astropy.table.Column,
            Represents a derived column (see below)

    Derived columns
    ---------------
        _x, _y, _z : derived from __init__, 
            position from the center (e.g. _x = x - x_center).
        _r : derived from __init__, 
            distance from the center (projected or intrinsic)
        _mask : derived from __init__, 
            boolean mask used to filter the snapshot
        _rho : derived from add_local_density,
            local density
    
    Methods
    -------
        filter(overwrite=True, by_range=None, **kwargs)
        add_local_density(n_neighbors=6)

    Notes
    -----
    N, project and center are properties of the class that can be set with 
    consistent setter methods. Namely, changing values for these properties
    has an effect on other quantities of the snapshot. 
    In particular, 
    N = val : permanently truncate the orginal snapshot to the first val elements
    center = (...) : re-center the cluster by changing _x,_y,_z,_r.
    project = True/False : re-calculate _r 
    """

    def __init__(self, data=None, names=None, dtype=None, 
        project=False, center=(0,0,0)):
        self._data = Table(data=data, names=names, dtype=dtype)

        self._check_features('x', 'y', 'z', function_id='Initialization')
        self._colnames_init = self._data.colnames
        self.__project = project # initialize project
        self.__center = center # initialize center
        self.center = center # use center setter to initialize also _x, _y, _z, _r
        self.__N = len(self._data) # initialize number of stars (do not use N.setter)
        self._data['_mask'] = np.ones(self.N, dtype=bool)
        
    def _check_features(self, *args, **kwargs):
        """
        Raise a ValueError if one of the features specified by args are 
        not column names of the cluster

        args : str,
            Column names
        kwargs : {function_id : str},
            Calling function 
        """
        # default, cannnot trace back to the calling function 
        kwargs["function_id"] = kwargs.get("function_id", "?? ") 

        missing_features = []
        for feature in args:
            if feature not in self._data.colnames:
                missing_features.append(feature)
        if len(missing_features) > 0:
            raise ValueError(kwargs["function_id"] + ': provide ' + \
                str(missing_features))
        
    def filter(self, overwrite=True, by_range=None, **kwargs):
        """
        Mask Snapshot by selecting stars according to different
        criteria. All the criteria specified are combined in a logical 
        union mask.
        If by range is True, star are selected with the provided ranges. 
        Otherwise, only stars with matching values are selected.
        If overwrite is True, the masking is applied to the original snapshot. 
        Otherwise, it is applied to the existent filter.

        Parameters
        ----------
            overwrite : bool
                Specify if the filter has to be overwritten to the existent one.
            by_range : dict, optional
                Specify filter types. Keys are Snapshot columns and values are
                boolean values. Default is true for all filters. 
            kwargs : keys are Snapshot columns and values are list-like objects   

        Examples
        --------
            Select only main sequence stars with kw = 0 or 1:
            >> filter(by_range={'kw':False}, kw=(0,1))

            Select stars with radius within rmin and rmax:
            >> filter(_r=(rmin, rmax))

            Select both main sequence stars (kw = 0, 1) and stars within radius 10.:
            >> filter(by_range={'kw':False, '_r':True}, kw=(0,1), _r=(0.,10.))
        """
        # check if the number of elements in by_range matches the number of
        # kwargs
        if by_range is None: # default case
            by_range = {}
            for key in kwargs.keys():
                by_range[key] = True
        else: 
            if not (by_range.keys() == kwargs.keys()):
                raise ValueError("Invalid Filter")

        # check for invalid keys
        for key, value in kwargs.iteritems():
            if key not in self._data.colnames:
                raise KeyError(key + " is an invalid argument")

        # reset mask
        if (overwrite) or (not "_mask" in self._data.colnames): # reset masking
            self._data["_mask"] = np.ones(len(self._data), dtype='bool')

        self._data["_mask"] = self._get_mask(by_range, **kwargs)

    def _get_mask(self, by_range, **kwargs):
        """
        Return a single bool array, which is the logical union of
        all the mask criteria
        """
        # list of masking criteria: first element is the existent mask
        mask = [self._data["_mask"], ]

        for n, (key, value) in enumerate(kwargs.iteritems()):
            # for each masking criterium

            if not by_range[key]: 
                # select only elements in the list provided with value
                mask.append(np.in1d(self._data[key], value)) 
            else: 
                # select only elements in the given range
                try:
                    if (len(value) == 2):
                        condition = np.logical_and(
                            self._data[key] >= value[0],
                            self._data[key] < value[1]
                            )
                    else: # raise a sure error to exit with ValueError
                        tmp = value[len(value)] # dummy line
                except:
                    raise ValueError(
                        str(value) + " is an invalid value for masking " \
                        + key)
                mask.append(condition)
        if len(mask) == 1:
            return mask[0]
        if len(mask) > 1:
            return np.logical_and.reduce(mask)

    def add_local_density(self, n_neighbors=6):
        """
        Add hidden column for local density (relies on gcp.get_local_density). 

        Parameter
        ---------
            n_neighbors : int, optional
                Number of neighbors for local density
        Notes
        -----
            Local density is calculated for the subgroup of selected stars.
        """
        if hasattr(self._data, "_rho"):
            return
        rho = gcp.get_local_density(self, 
            n_neighbors=n_neighbors, masked=True)
        self._data["_rho"] = np.full(len(self._data), np.nan)
        self._data["_rho"][self._data['_mask']] = rho

    def sort(self, column):
        """
        Sort Snapshot according to the specified column.

        Parameter
        ---------
            column : str,
                Column name of the quantity used for argsorting
        """
        # check if column exists
        self._check_features(column, function_id='argsort')
        self._data = self._data[np.argsort(self._data[column])]

    @property
    def N(self):
        return self.__N

    @N.setter
    def N(self, val):
        assert isinstance(val, int), "N must be an integer value"
        assert val > 0 and val < self.N, "Invalid value for N"
        self.__N = val
        truncate = np.ones(len(self._data), dtype=bool)
        truncate[val:] = False
        self._data = self._data[truncate]

    @property
    def center(self):
        return self.__center

    @center.setter   
    def center(self, center):
        assert len(center) == 3, "Specify 3 coordinates for the new center!"
        self.__center = center
        self._data["_x"] = self._data["x"] - center[0]
        self._data["_y"] = self._data["y"] - center[1]
        self._data["_z"] = self._data["z"] - center[2]
        self._calculate_radius()

    @property
    def project(self):
        return self.__project

    @project.setter
    def project(self, val):
        assert isinstance(val, bool), "project must be True or False"
        self.__project = val
        self._calculate_radius()
        
    @property
    def original(self):
        """ Original snapshot. """
        return self._data[self._colnames_init]

    @property
    def rad_sorted(self):
        """ Masked Snapshot, radially sorted. """
        so = np.argsort(self._data['_r'])
        return self._data[self._colnames_init][so]
        
    def __getitem__(self, item):
        return self._data[self._data['_mask']][item]
    
    def _calculate_radius(self):
        if self.project:
            self._data["_r"] = np.sqrt(self._data["_x"]**2. + self._data["_y"]**2.)
        else:
            self._data["_r"] = np.sqrt(
                self._data["_x"]**2. + self._data["_y"]**2. + self._data["_z"]**2.)

    def __repr__(self):
        return "Snapshot : N = %i, project = %r, center = (%.1f, %.1f, %.1f)" % \
            (self.N, self.project, self.center[0], self.center[1], self.center[2])

    def __str__(self):
        return '\n'.join(self[:].pformat())