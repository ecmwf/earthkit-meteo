# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


"""
Lunar computation functions.

The API is split into two layers:

- Low-level implementations are in the ``array`` submodules.
- High-level functions are in this module and dispatch to backend implementations
  based on input type.

Currently, the only supported backend is the ``array`` implementation, so the low-level and high-level functions are identical. In the future, additional backends may be added, such as ``xarray`` and ``fieldlist``.
"""

from .lunar import *  # noqa

__all__ = [
    "distance_from_earth_centre_to_moon",
    "distance_to_moon",
    "delta_distance_to_moon",
]
