# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""
Thermo related functions operating on xarray objects.
"""

from .thermo import celsius_to_kelvin
from .thermo import dewpoint_from_relative_humidity
from .thermo import dewpoint_from_specific_humidity
from .thermo import ept_from_dewpoint
from .thermo import ept_from_specific_humidity
from .thermo import kelvin_to_celsius
from .thermo import lcl
from .thermo import lcl_temperature
from .thermo import mixing_ratio_from_dewpoint
from .thermo import mixing_ratio_from_specific_humidity
from .thermo import mixing_ratio_from_vapour_pressure
from .thermo import potential_temperature
from .thermo import pressure_on_dry_adiabat
from .thermo import relative_humidity_from_dewpoint
from .thermo import relative_humidity_from_specific_humidity
from .thermo import saturation_ept
from .thermo import saturation_mixing_ratio
from .thermo import saturation_mixing_ratio_slope
from .thermo import saturation_specific_humidity
from .thermo import saturation_specific_humidity_slope
from .thermo import saturation_vapour_pressure
from .thermo import saturation_vapour_pressure_slope
from .thermo import specific_gas_constant
from .thermo import specific_humidity_from_dewpoint
from .thermo import specific_humidity_from_mixing_ratio
from .thermo import specific_humidity_from_relative_humidity
from .thermo import specific_humidity_from_vapour_pressure
from .thermo import temperature_from_potential_temperature
from .thermo import temperature_from_saturation_vapour_pressure
from .thermo import temperature_on_dry_adiabat
from .thermo import temperature_on_moist_adiabat
from .thermo import vapour_pressure_from_mixing_ratio
from .thermo import vapour_pressure_from_specific_humidity
from .thermo import virtual_potential_temperature
from .thermo import virtual_temperature
from .thermo import wet_bulb_potential_temperature_from_dewpoint
from .thermo import wet_bulb_potential_temperature_from_specific_humidity
from .thermo import wet_bulb_temperature_from_dewpoint
from .thermo import wet_bulb_temperature_from_specific_humidity

__all__ = [
    "celsius_to_kelvin",
    "dewpoint_from_relative_humidity",
    "dewpoint_from_specific_humidity",
    "ept_from_dewpoint",
    "ept_from_specific_humidity",
    "kelvin_to_celsius",
    "lcl",
    "lcl_temperature",
    "mixing_ratio_from_dewpoint",
    "mixing_ratio_from_specific_humidity",
    "mixing_ratio_from_vapour_pressure",
    "potential_temperature",
    "pressure_on_dry_adiabat",
    "relative_humidity_from_dewpoint",
    "relative_humidity_from_specific_humidity",
    "saturation_ept",
    "saturation_mixing_ratio",
    "saturation_mixing_ratio_slope",
    "saturation_specific_humidity",
    "saturation_specific_humidity_slope",
    "saturation_vapour_pressure",
    "saturation_vapour_pressure_slope",
    "specific_gas_constant",
    "specific_humidity_from_dewpoint",
    "specific_humidity_from_mixing_ratio",
    "specific_humidity_from_relative_humidity",
    "specific_humidity_from_vapour_pressure",
    "temperature_from_potential_temperature",
    "temperature_from_saturation_vapour_pressure",
    "temperature_on_dry_adiabat",
    "temperature_on_moist_adiabat",
    "vapour_pressure_from_mixing_ratio",
    "vapour_pressure_from_specific_humidity",
    "virtual_potential_temperature",
    "virtual_temperature",
    "wet_bulb_potential_temperature_from_dewpoint",
    "wet_bulb_potential_temperature_from_specific_humidity",
    "wet_bulb_temperature_from_dewpoint",
    "wet_bulb_temperature_from_specific_humidity",
]
