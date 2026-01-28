# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from functools import wraps
from typing import Any
import xarray as xr

import numpy as np

from .. import array

# from earthkit.meteo.utils.metadata import metadata_handler
METADATA_DEFAULTS: dict[str, dict[str, object]] = {
    "mixing_ratio": {
        "standard_name": {"humidity_mixing_ratio"},
        "units": {"kg kg-1", "kg/kg"},
        "long_name": {
            "mixing ratio",
            "water vapor mixing ratio",
        },
    },
    "specific_humidity": {
        "standard_name": {"specific_humidity"},
        "units": {"kg kg-1", "kg/kg"},
        "long_name": {
            "specific humidity",
        },
    },
    "temperature": {
        "standard_name": {"air_temperature"},
        "units": {"K", "kelvin", "Celsius", "C"},
        "long_name": {
            "air temperature",
        },
    },
}

def metadata_handler(inputs: list[str] | dict[str, Any], outputs: list[str] | dict[str, Any]):
    """
    Decorator to check and update CF metadata on inputs and outputs of a function.
    
    - Inputs: validates that inputs conform to CF attributes.
    - Outputs: fills in or corrects metadata according to CF metadata dictionary.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # --- check inputs ---
            for var_name, arg in zip(inputs, args):
                if var_name not in METADATA_DEFAULTS:
                    continue
                allowed = METADATA_DEFAULTS[var_name]
                if isinstance(arg, xr.DataArray):
                    # validate units & attributes
                    _validate_and_warn(arg, allowed)

            # --- compute outputs ---
            result = func(*args, **kwargs)

            # normalize outputs to tuple for uniform handling
            outputs_list = result if isinstance(result, tuple) else (result,)
            updated_outputs = []

            # --- update outputs metadata ---
            for var_name, out in zip(outputs, outputs_list):
                if var_name not in METADATA_DEFAULTS:
                    updated_outputs.append(out)
                    continue
                allowed = METADATA_DEFAULTS[var_name]
                if isinstance(outputs, dict):
                    allowed |= outputs[var_name]
                if isinstance(out, xr.DataArray):
                    out = _update_metadata(out, allowed)
                updated_outputs.append(out)

            # return single object or tuple
            if isinstance(result, tuple):
                return tuple(updated_outputs)
            return updated_outputs[0]

        return wrapper

    return decorator

def _validate_and_warn(da: xr.DataArray, allowed_attrs: dict):
    """
    Validate the DataArray attributes. Raise a warning if they do not match.
    """
    import warnings

    # Check units
    if "units" in allowed_attrs and da.attrs.get("units") not in allowed_attrs["units"]:
        warnings.warn(
            f"{da.name} units '{da.attrs.get('units')}' do not match allowed '{allowed_attrs['units']}'"
        )

    # Check other attributes
    for key, val in allowed_attrs.items():
        if key == "units":
            continue
        if da.attrs.get(key).lower() not in val:
            warnings.warn(
                f"{da.name} attribute '{key}'='{da.attrs.get(key)}' "
                f"does not match allowed '{val}'"
            )


def _update_metadata(da: xr.DataArray, allowed_attrs: dict) -> xr.DataArray:
    """
    Update the DataArray attributes to match CF metadata.
    """
    for key, val in allowed_attrs.items():
        val = list(val)[0] if isinstance(val, set) else val
        da.attrs[key] = val
    return da


def _sample_arg(arg: object) -> np.ndarray:
    if hasattr(arg, "dtype"):
        return np.zeros((), dtype=arg.dtype)
    return np.zeros((), dtype=float)


def _infer_output_dtypes(func, *args, **kwargs) -> list[np.dtype]:
    sample_args = [_sample_arg(arg) for arg in args]
    res = func(*sample_args, **kwargs)
    if isinstance(res, tuple):
        return [np.asarray(item).dtype for item in res]
    return [np.asarray(res).dtype]


def _apply_ufunc(func, *args, **kwargs):
    import xarray as xr

    output_dtypes = _infer_output_dtypes(func, *args, **kwargs)

    return xr.apply_ufunc(
        func,
        *args,
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=output_dtypes,
        keep_attrs=True,
    )

def celsius_to_kelvin(t):
    r"""Convert temperature values from Celsius to Kelvin.

    Parameters
    ----------
    t : xarray.DataArray
        Temperature in Celsius units

    Returns
    -------
    xarray.DataArray
        Temperature in Kelvin units

    """
    return _apply_ufunc(array.celsius_to_kelvin, t)

def kelvin_to_celsius(t):
    r"""Convert temperature values from Kelvin to Celsius.

    Parameters
    ----------
    t : xarray.DataArray
        Temperature in Kelvin units

    Returns
    -------
    xarray.DataArray
        Temperature in Celsius units

    """
    return _apply_ufunc(array.kelvin_to_celsius, t)


@metadata_handler(inputs=["mixing_ratio"], outputs=["specific_humidity"])
def specific_humidity_from_mixing_ratio(w, t):
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
    res = _apply_ufunc(array.specific_humidity_from_mixing_ratio, w)
    res.name = "specific_humidity"
    res.attrs["standard_name"] = "specific_humidity"
    res.attrs["long_name"] = "Specific Humidity"
    res.attrs["units"] = "kg/kg"
    return res

def relative_humidity_from_dewpoint(t, td):
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
    res = _apply_ufunc(array.relative_humidity_from_dewpoint, t, td)
    res.name = "relative_humidity"
    res.attrs["standard_name"] = "relative_humidity"
    res.attrs["long_name"] = "Relative Humidity"
    res.attrs["units"] = "%"
    return res

def relative_humidity_from_specific_humidity(t, q, p):
    r"""Compute the relative humidity from specific humidity.

    Parameters
    ----------
    t: array-like
        Temperature (K)
    q: array-like
        Specific humidity (kg/kg)
    p: array-like
        Pressure (Pa)

    Returns
    -------
    array-like
        Relative humidity (%)


    The computation is based on the following formula:

    .. math::

        r = 100 \frac {e(q, p)}{e_{msat}(t)}

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure_from_specific_humidity`)
        * :math:`e_{msat}` is the :func:`saturation_vapour_pressure` based on the "mixed" phase

    """
    res = _apply_ufunc(array.relative_humidity_from_specific_humidity, t, q, p)
    res.name = "relative_humidity"
    res.attrs["standard_name"] = "relative_humidity"
    res.attrs["long_name"] = "Relative Humidity"
    res.attrs["units"] = "%"
    return res
