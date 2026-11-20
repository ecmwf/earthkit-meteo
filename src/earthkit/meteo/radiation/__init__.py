# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""
Surface radiation functions.

The API is split into two layers:

- Low-level implementations are in the ``array``, ``xarray`` and ``fieldlist`` submodules.
- High-level functions are in this module and dispatch to backend implementations
  based on input type.
"""

from .radiation import surface_downward_shortwave_radiation, surface_downward_longwave_flux

__all__ = [
    "surface_downward_shortwave_radiation",
    "surface_downward_longwave_flux",
]
