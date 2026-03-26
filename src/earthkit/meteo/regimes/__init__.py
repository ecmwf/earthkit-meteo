# (C) Copyright 2025- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""
Weather regimes based on projections onto spatial patterns.

- To define a collection or generator of patterns, use the provided generator
  classes to implement your desired scheme or define your own pattern generator
  scheme based on the abstract base class :py:class:`Patterns`.
- To compute regime indices, use the functions :py:func:`project` and
  :py:func:`regime_index` together with a given pattern collection/generator.


.. note::
    At the moment, only regular lat-lon grids are supported for the
    specification of patterns::

        {
            "grid": [lon_spacing, lat_spacing],
            "area": [lat0, lon0, lat1, lon1]
        }
"""

from . import array
from .patterns import ConstantPatterns
from .patterns import ModulatedPatterns
from .patterns import Patterns

# Offer xarray implementations at high-level (TODO: support fieldlist)
from .xarray import project
from .xarray import regime_index
