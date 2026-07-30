"""
Solar computation functions.

The API is split into two layers:

- Low-level implementations are in the ``array``, ``xarray`` and ``fieldlist`` submodules.
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
