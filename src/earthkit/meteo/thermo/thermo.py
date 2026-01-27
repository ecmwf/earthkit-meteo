# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from typing import overload, Any, TypeVar

import xarray as xr
import earthkit.data as ekd

from . import array


FieldType = TypeVar("FieldType", ekd.Field, ekd.FieldList)

@overload
def celsius_to_kelvin(
    t: xr.DataArray,
    *,
    some_xarray_specific_option: Any = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Convert temperature from Celsius to Kelvin for xarray.DataArray."""
    ...


@overload
def celsius_to_kelvin(
    t: FieldType,
    *,
    some_grib_specific_option: Any = None,
    **kwargs: Any,
) -> FieldType:
    """Convert temperature from Celsius to Kelvin for earthkit objects."""
    ...


# TODO 1: CF conventions through Enums?
def celsius_to_kelvin(t, **kwargs: Any) -> Any:
    """Convert temperature from Celsius to Kelvin."""

    UNITS = ["celsius", "degree_celsius", "degrees_celsius", "°c"]
    if isinstance(t, xr.DataArray):

        # check variable
        if "standard_name" in t.attrs:
            if t.attrs["standard_name"] != "air_temperature":
                raise ValueError(
                    f"Expected 'standard_name' attribute to be 'air_temperature', got '{t.attrs['standard_name']}'"
                )
        elif "long_name" in t.attrs:
            if "temperature" not in t.attrs["long_name"].lower():
                raise ValueError(
                    f"Expected 'long_name' attribute to contain 'temperature', got '{t.attrs['long_name']}'"
                )
        else:
            raise ValueError("DataArray must have either 'standard_name' or 'long_name' attribute to convert temperature.")

        # check units 
        if "units" not in t.attrs:
            raise ValueError("DataArray must have 'units' attribute to convert temperature.")
        if t.attrs["units"].lower() not in UNITS:
            msg = f"Expected 'units' attribute to be one of {UNITS}, got '{t.attrs['units']}'"
            raise ValueError(msg)

        t = xr.apply_ufunc(array.celsius_to_kelvin, t)
        t.attrs["units"] = "Kelvin"
        return t
    
    return array.celsius_to_kelvin(t, **kwargs)


@overload
def kelvin_to_celsius(
    t: xr.DataArray,
    *,
    some_xarray_specific_option: Any = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Convert temperature from Kelvin to Celsius for xarray.DataArray."""
    ...

@overload
def kelvin_to_celsius(
    t: FieldType,
    *,
    some_grib_specific_option: Any = None,
    **kwargs: Any,
) -> FieldType:
    """Convert temperature from Kelvin to Celsius for earthkit objects."""
    ...

def kelvin_to_celsius(t, **kwargs):
    """Convert temperature from Kelvin to Celsius."""

    if isinstance(t, xr.DataArray):

        # check variable
        if "standard_name" in t.attrs:
            if t.attrs["standard_name"] != "air_temperature":
                raise ValueError(
                    f"Expected 'standard_name' attribute to be 'air_temperature', got '{t.attrs['standard_name']}'"
                )
        elif "long_name" in t.attrs:
            if "temperature" not in t.attrs["long_name"].lower():
                raise ValueError(
                    f"Expected 'long_name' attribute to contain 'temperature', got '{t.attrs['long_name']}'"
                )
        else:
            raise ValueError("DataArray must have either 'standard_name' or 'long_name' attribute to convert temperature.")

        # check units 
        if "units" not in t.attrs:
            raise ValueError("DataArray must have 'units' attribute to convert temperature.")
        if t.attrs["units"].lower() != "kelvin":
            msg = f"Expected 'units' attribute to be 'Kelvin', got '{t.attrs['units']}'"
            raise ValueError(msg)

        t = xr.apply_ufunc(array.kelvin_to_celsius, t)
        t.attrs["units"] = "Celsius"
        return t
    
    return array.kelvin_to_celsius(t, **kwargs)
    


@overload
def specific_humidity_from_mixing_ratio(
    w: xr.DataArray,
    *,
    some_xarray_specific_option: Any = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Compute the specific humidity from mixing ratio for xarray.DataArray."""
    ...


@overload
def specific_humidity_from_mixing_ratio(
    w: FieldType,
    *,
    some_grib_specific_option: Any = None,
    **kwargs: Any,
) -> FieldType:
    """Compute the specific humidity from mixing ratio for earthkit objects."""
    ...

def specific_humidity_from_mixing_ratio(w, **kwargs):
    UNITS = ["1", "Kg/Kg"] 
    if isinstance(w, xr.DataArray):
        # check variable
        if "standard_name" in w.attrs:
            if w.attrs["standard_name"] != "humidity_mixing_ratio":
                raise ValueError(
                    f"Expected 'standard_name' attribute to be 'humidity_mixing_ratio', got '{w.attrs['standard_name']}'"
                )
        elif "long_name" in w.attrs:
            if "temperature" not in w.attrs["long_name"].lower():
                raise ValueError(
                    f"Expected 'long_name' attribute to contain 'temperature', got '{w.attrs['long_name']}'"
                )
        else:
            raise ValueError("DataArray must have either 'standard_name' or 'long_name' attribute to compute the specific humidity.")
        
        # check units 
        if "units" not in w.attrs:
            raise ValueError("DataArray must have 'units' attribute to compute the specific humidity.")
        if w.attrs["units"].lower() not in UNITS:
            msg = f"Expected 'units' attribute to be one of {UNITS}, got '{w.attrs['units']}'"
            raise ValueError(msg)
        
        q = xr.apply_ufunc(array.specific_humidity_from_mixing_ratio, w)
        q.attrs["standard_name"] = "specific_humidity"
        return q

    return array.specific_humidity_from_mixing_ratio(w, **kwargs)


def mixing_ratio_from_specific_humidity(*args, **kwargs):
    return array.mixing_ratio_from_specific_humidity(*args, **kwargs)


def vapour_pressure_from_specific_humidity(*args, **kwargs):
    return array.vapour_pressure_from_specific_humidity(*args, **kwargs)


def vapour_pressure_from_mixing_ratio(*args, **kwargs):
    return array.vapour_pressure_from_mixing_ratio(*args, **kwargs)


def specific_humidity_from_vapour_pressure(*args, **kwargs):
    return array.specific_humidity_from_vapour_pressure(*args, **kwargs)


def mixing_ratio_from_vapour_pressure(*args, **kwargs):
    return array.mixing_ratio_from_vapour_pressure(*args, **kwargs)


def saturation_vapour_pressure(*args, **kwargs):
    return array.saturation_vapour_pressure(*args, **kwargs)


def saturation_mixing_ratio(*args, **kwargs):
    return array.saturation_mixing_ratio(*args, **kwargs)


def saturation_specific_humidity(*args, **kwargs):
    return array.saturation_specific_humidity(*args, **kwargs)


def saturation_vapour_pressure_slope(*args, **kwargs):
    return array.saturation_vapour_pressure_slope(*args, **kwargs)


def saturation_mixing_ratio_slope(*args, **kwargs):
    return array.saturation_mixing_ratio_slope(*args, **kwargs)


def saturation_specific_humidity_slope(*args, **kwargs):
    return array.saturation_specific_humidity_slope(*args, **kwargs)


def temperature_from_saturation_vapour_pressure(*args, **kwargs):
    return array.temperature_from_saturation_vapour_pressure(*args, **kwargs)


def relative_humidity_from_dewpoint(*args, **kwargs):
    return array.relative_humidity_from_dewpoint(*args, **kwargs)


def relative_humidity_from_specific_humidity(*args, **kwargs):
    return array.relative_humidity_from_specific_humidity(*args, **kwargs)


def specific_humidity_from_dewpoint(*args, **kwargs):
    return array.specific_humidity_from_dewpoint(*args, **kwargs)


def mixing_ratio_from_dewpoint(*args, **kwargs):
    return array.mixing_ratio_from_dewpoint(*args, **kwargs)


def specific_humidity_from_relative_humidity(*args, **kwargs):
    return array.specific_humidity_from_relative_humidity(*args, **kwargs)


def dewpoint_from_relative_humidity(*args, **kwargs):
    return array.dewpoint_from_relative_humidity(*args, **kwargs)


def dewpoint_from_specific_humidity(*args, **kwargs):
    return array.dewpoint_from_specific_humidity(*args, **kwargs)


def virtual_temperature(*args, **kwargs):
    return array.virtual_temperature(*args, **kwargs)


def virtual_potential_temperature(*args, **kwargs):
    return array.virtual_potential_temperature(*args, **kwargs)


def potential_temperature(*args, **kwargs):
    return array.potential_temperature(*args, **kwargs)


def temperature_from_potential_temperature(*args, **kwargs):
    return array.temperature_from_potential_temperature(*args, **kwargs)


def pressure_on_dry_adiabat(*args, **kwargs):
    return array.pressure_on_dry_adiabat(*args, **kwargs)


def temperature_on_dry_adiabat(*args, **kwargs):
    return array.temperature_on_dry_adiabat(*args, **kwargs)


def lcl_temperature(*args, **kwargs):
    return array.lcl_temperature(*args, **kwargs)


def lcl(*args, **kwargs):
    return array.lcl(*args, **kwargs)


def ept_from_dewpoint(*args, **kwargs):
    return array.ept_from_dewpoint(*args, **kwargs)


def ept_from_specific_humidity(*args, **kwargs):
    return array.ept_from_specific_humidity(*args, **kwargs)


def saturation_ept(*args, **kwargs):
    return array.saturation_ept(*args, **kwargs)


def temperature_on_moist_adiabat(*args, **kwargs):
    return array.temperature_on_moist_adiabat(*args, **kwargs)


def wet_bulb_temperature_from_dewpoint(*args, **kwargs):
    return array.wet_bulb_temperature_from_dewpoint(*args, **kwargs)


def wet_bulb_temperature_from_specific_humidity(*args, **kwargs):
    return array.wet_bulb_temperature_from_specific_humidity(*args, **kwargs)


def wet_bulb_potential_temperature_from_dewpoint(*args, **kwargs):
    return array.wet_bulb_potential_temperature_from_dewpoint(*args, **kwargs)


def wet_bulb_potential_temperature_from_specific_humidity(*args, **kwargs):
    return array.wet_bulb_potential_temperature_from_specific_humidity(*args, **kwargs)


def specific_gas_constant(*args, **kwargs):
    return array.specific_gas_constant(*args, **kwargs)
 