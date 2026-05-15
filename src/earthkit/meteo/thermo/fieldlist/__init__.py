# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""
Thermodynamic related functions operating on earthkit.data FieldList objects.
"""

from .thermo import *  # noqa

# from earthkit.meteo.thermo.fieldlist.thermo import (
#     dewpoint_from_relative_humidity,
#     dewpoint_from_specific_humidity,
#     ept_from_dewpoint,
#     ept_from_specific_humidity,
#     lcl,
#     lcl_temperature,
#     mixing_ratio_from_dewpoint,
#     mixing_ratio_from_specific_humidity,
#     mixing_ratio_from_vapour_pressure,
#     potential_temperature,
#     pressure_on_dry_adiabat,
#     relative_humidity_from_dewpoint,
#     relative_humidity_from_specific_humidity,
#     saturation_ept,
#     saturation_mixing_ratio,
#     saturation_mixing_ratio_slope,
#     saturation_specific_humidity,
#     saturation_specific_humidity_slope,
#     saturation_vapour_pressure,
#     saturation_vapour_pressure_slope,
#     specific_gas_constant,
#     specific_humidity_from_dewpoint,
#     specific_humidity_from_mixing_ratio,
#     specific_humidity_from_relative_humidity,
#     specific_humidity_from_vapour_pressure,
#     temperature_from_potential_temperature,
#     temperature_from_saturation_vapour_pressure,
#     temperature_on_dry_adiabat,
#     temperature_on_moist_adiabat,
#     vapour_pressure_from_mixing_ratio,
#     vapour_pressure_from_specific_humidity,
#     virtual_potential_temperature,
#     virtual_temperature,
#     wet_bulb_potential_temperature_from_dewpoint,
#     wet_bulb_potential_temperature_from_specific_humidity,
#     wet_bulb_temperature_from_dewpoint,
#     wet_bulb_temperature_from_specific_humidity,
# )

# __all__ = [
#     "dewpoint_from_relative_humidity",
#     "dewpoint_from_specific_humidity",
#     "ept_from_dewpoint",
#     "ept_from_specific_humidity",
#     "lcl",
#     "lcl_temperature",
#     "mixing_ratio_from_dewpoint",
#     "mixing_ratio_from_specific_humidity",
#     "mixing_ratio_from_vapour_pressure",
#     "potential_temperature",
#     "pressure_on_dry_adiabat",
#     "relative_humidity_from_dewpoint",
#     "relative_humidity_from_specific_humidity",
#     "saturation_ept",
#     "saturation_mixing_ratio",
#     "saturation_mixing_ratio_slope",
#     "saturation_specific_humidity",
#     "saturation_specific_humidity_slope",
#     "saturation_vapour_pressure",
#     "saturation_vapour_pressure_slope",
#     "specific_gas_constant",
#     "specific_humidity_from_dewpoint",
#     "specific_humidity_from_mixing_ratio",
#     "specific_humidity_from_relative_humidity",
#     "specific_humidity_from_vapour_pressure",
#     "temperature_from_potential_temperature",
#     "temperature_from_saturation_vapour_pressure",
#     "temperature_on_dry_adiabat",
#     "temperature_on_moist_adiabat",
#     "vapour_pressure_from_mixing_ratio",
#     "vapour_pressure_from_specific_humidity",
#     "virtual_potential_temperature",
#     "virtual_temperature",
#     "wet_bulb_potential_temperature_from_dewpoint",
#     "wet_bulb_potential_temperature_from_specific_humidity",
#     "wet_bulb_temperature_from_dewpoint",
#     "wet_bulb_temperature_from_specific_humidity",
# ]
