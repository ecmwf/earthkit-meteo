"""Solar computation functions operating on xarray objects."""

from .solar import (
    cos_solar_zenith_angle,
    cos_solar_zenith_angle_integrated,
    incoming_solar_radiation,
    julian_day,
    solar_declination_angle,
    toa_incident_solar_radiation,
)

__all__ = [
    "cos_solar_zenith_angle",
    "cos_solar_zenith_angle_integrated",
    "incoming_solar_radiation",
    "julian_day",
    "solar_declination_angle",
    "toa_incident_solar_radiation",
]
