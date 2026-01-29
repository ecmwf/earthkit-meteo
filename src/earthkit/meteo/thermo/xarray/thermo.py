# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np

from earthkit.meteo.utils.decorators import metadata_handler

from .. import array


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


@metadata_handler(inputs=["temperature", "dewpoint"], outputs=["relative_humidity"])
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
    return _apply_ufunc(array.relative_humidity_from_dewpoint, t, td)


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
