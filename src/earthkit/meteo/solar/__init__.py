# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


"""
Solar computation functions.

The API is split into two layers:

- Low-level implementations are in the ``array`` and ``xarray`` submodules.
- High-level functions are in this module and dispatch to backend implementations
  based on input type.
"""

from .solar import *  # noqa

__all__ = [
    "julian_day",
    "solar_declination_angle",
    "cos_solar_zenith_angle",
    "cos_solar_zenith_angle_integrated",
    "incoming_solar_radiation",
    "toa_incident_solar_radiation",
]
