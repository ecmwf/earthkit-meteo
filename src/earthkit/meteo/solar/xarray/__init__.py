# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Solar computation functions operating on xarray objects."""

from .solar import cos_solar_zenith_angle
from .solar import cos_solar_zenith_angle_integrated
from .solar import incoming_solar_radiation
from .solar import julian_day
from .solar import solar_declination_angle
from .solar import toa_incident_solar_radiation

__all__ = [
    "cos_solar_zenith_angle",
    "cos_solar_zenith_angle_integrated",
    "incoming_solar_radiation",
    "julian_day",
    "solar_declination_angle",
    "toa_incident_solar_radiation",
]
