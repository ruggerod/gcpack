# gcpack

gcpack is a Python module made for astronomers. It offers an intuitive support for the analysis of the internal dynamics of globular clusters.

## Getting Started

This module is still under development. To use it, please fork the project and dowload a local copy.

### Prerequisites

gcpack requires:
* python (2.7)
* numpy
* astropy
* scikit-learn

## Running

The fundamental class of the module is Snapshot, which represents the stellar system.
A Snapshot, may be randomized (i.e. a rotation of its coordinates is performed), 
projected along the z axis or centered in a particular position.
Moreover, a subgroup of the cluster may be selected.

```
import gcpack as gcp
import numpy as np

# Create an unphysical snapshot with 4 features, i.e. space coordinates and mass.
snap = gcp.Snapshot(data=np.random.rand(10, 4),
                    names=['x', 'y', 'z', 'm'],
                    project=False,
                    center=[0, 0, 0])

# mask only stars within a radius of 0.5
snap.filter(_r=(0., 0.5), by_range={"_r": True})
```

Some basic operations can be done on a Snapshot, such as calculating its center of mass:

```
gcp.center_of_mass(snap, masked=True)
```

or producing a velocity dispersion profile:

```
gcp.velocity_dispersion_profile(snap,
                                radial_bins=[[0.,0.3], [0.3,0.6], [0.6, 0.9]],
                                los=False
                                )
```

## Author

* [**Ruggero de Vita**](https://github.com/ruggerod)