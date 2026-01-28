# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any  # noqa: F401
from typing import Iterable
from typing import overload

if TYPE_CHECKING:
    import xarray  # type: ignore[import]
    from earthkit.data import FieldList  # type: ignore[import]

# TODO: move these underscore functions to meteo.utils
def _is_xarray(obj: Any) -> bool:
    from earthkit.meteo.utils import is_module_loaded

    if not is_module_loaded("xarray"):
        return False

    try:
        import xarray as xr

        return isinstance(obj, (xr.DataArray, xr.Dataset))
    except (ImportError, RuntimeError, SyntaxError):
        return False


def _is_fieldlist(obj: Any) -> bool:
    from earthkit.meteo.utils import is_module_loaded

    if not is_module_loaded("earthkit.data"):
        return False

    try:
        from earthkit.data import FieldList

        return isinstance(obj, FieldList)
    except ImportError:
        return False


def _call(func: str, *args: Any, **kwargs: Any) -> Any:
    if _is_xarray(args[0]):
        from . import xarray as _module
    elif _is_fieldlist(args[0]):
        from . import fieldlist as _module

    return getattr(_module, func)(*args, **kwargs)




@overload
def celsius_to_kelvin(
    t: xr.DataArray,
    **kwargs: Any,
) -> xr.DataArray:
    ...


@overload
def celsius_to_kelvin(
    t: FieldType,
    **kwargs: Any,
) -> FieldType:
    ...


# TODO 1: CF conventions through Enums?
def celsius_to_kelvin(t, **kwargs: Any) -> Any:
    """Convert temperature from Celsius to Kelvin."""

    UNITS = ["celsius", "degree_celsius", "degrees_celsius", "°c"]
    if _is_xarray(t):

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

    if _is_xarray(t):

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
def specific_humidity_from_mixing_ratio(w: "xarray.DataArray") -> "xarray.DataArray":
    r"""Compute the specific humidity from mixing ratio.

    Parameters
    ----------
    w : xarray.DataArray
        Mixing ratio (kg/kg)

    Returns
    -------
    xarray.DataArray
        Specific humidity (kg/kg)


    The result is the specific humidity in kg/kg units. The computation is based on
    the following definition [Wallace2006]_:

    .. math::

        q = \frac {w}{1+w}

    """
    ...

@overload
def specific_humidity_from_mixing_ratio(w: "FieldList") -> "FieldList":
    r"""Compute the specific humidity from mixing ratio.

    Parameters
    ----------
    w : "FieldList"
        Mixing ratio (kg/kg)

    Returns
    -------
    "FieldList"
        Specific humidity (kg/kg)


    The result is the specific humidity in kg/kg units. The computation is based on
    the following definition [Wallace2006]_:

    .. math::

        q = \frac {w}{1+w}

    """
    ...

def specific_humidity_from_mixing_ratio(w: Any) -> Any:
    r"""Compute the specific humidity from mixing ratio.

    Parameters
    ----------
    w : Any
        Mixing ratio (kg/kg)

    Returns
    -------
    Any
        Specific humidity (kg/kg)


    The result is the specific humidity in kg/kg units. The computation is based on
    the following definition [Wallace2006]_:

    .. math::

        q = \frac {w}{1+w}

    """
    return _call("specific_humidity_from_mixing_ratio", w)


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

@overload
def relative_humidity_from_dewpoint(t: "xarray.DataArray", td: "xarray.DataArray") -> "xarray.DataArray":
    r"""Compute the relative humidity from dew point temperature

    Parameters
    ----------
    t : "xarray.DataArray"
        Temperature (K)
    td: "xarray.DataArray"
        Dewpoint (K)


    Returns
    -------
    "xarray.DataArray"
        Relative humidity (%)


    The computation is based on the following formula:

    .. math::

        r = 100 \frac {e_{wsat}(td)}{e_{wsat}(t)}

    where :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water.

    """

@overload
def relative_humidity_from_dewpoint(t: "FieldList", td: "FieldList") -> "FieldList":
    r"""Compute the relative humidity from dew point temperature

    Parameters
    ----------
    t : "FieldList"
        Temperature (K)
    td: "FieldList"
        Dewpoint (K)


    Returns
    -------
    "FieldList"
        Relative humidity (%)


    The computation is based on the following formula:

    .. math::

        r = 100 \frac {e_{wsat}(td)}{e_{wsat}(t)}

    where :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water.

    """

def relative_humidity_from_dewpoint(t, td, **kwargs: Any) -> Any:
    r"""Compute the relative humidity from dew point temperature

    Parameters
    ----------
    t : Any
        Temperature (K)
    td: Any
        Dewpoint (K)


    Returns
    -------
    Any
        Relative humidity (%)


    The computation is based on the following formula:

    .. math::

        r = 100 \frac {e_{wsat}(td)}{e_{wsat}(t)}

    where :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water.

    """
    return _call("relative_humidity_from_dewpoint", t, td)

@overload
def relative_humidity_from_specific_humidity(
    t: xr.DataArray,
    *,
    some_xarray_specific_option: Any = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Compute the relative humidity from specific humidity for xarray.DataArray.."""
    ...

@overload
def relative_humidity_from_specific_humidity(
    t: FieldType,
    *,
    some_grib_specific_option: Any = None,
    **kwargs: Any,
) -> FieldType:
     """Compute the relative humidity from specific humidity for earthkit objects."""
     ...


def relative_humidity_from_specific_humidity(t, q, p, **kwargs):
    """Compute the relative humidity from specific humidity."""

    if isinstance(t, xr.DataArray) and isinstance(q, xr.DataArray) and isinstance(p, xr.DataArray):
        t = xr.apply_ufunc(array.relative_humidity_from_specific_humidity, t, q, p)
        t.attrs["units"] = "Percentage"
        return t

    return array.relative_humidity_from_specific_humidity(t, q, p, **kwargs)


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
