# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Solar computation functions operating on numpy arrays."""

from .solar import (
    DAYS_PER_YEAR,
    cos_solar_zenith_angle,
    cos_solar_zenith_angle_integrated,
    incoming_solar_radiation,
    julian_day,
    solar_declination_angle,
    toa_incident_solar_radiation,
)

__all__ = [
    "DAYS_PER_YEAR",
    "julian_day",
    "solar_declination_angle",
    "cos_solar_zenith_angle",
    "cos_solar_zenith_angle_integrated",
    "incoming_solar_radiation",
    "toa_incident_solar_radiation",
]
