# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import xarray as xr

from earthkit.meteo.utils.decorators import xarray_ufunc

from .. import array


def celsius_to_kelvin(t: xr.DataArray) -> xr.DataArray:
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
    return xarray_ufunc(array.celsius_to_kelvin, t).assign_attrs({
        "standard_name": "air_temperature",
        "units": "K",
    })



def kelvin_to_celsius(t: xr.DataArray) -> xr.DataArray:
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
    return xarray_ufunc(array.kelvin_to_celsius, t).assign_attrs({
        "standard_name": "air_temperature",
        "units": "degC",
    })


def specific_humidity_from_mixing_ratio(w: xr.DataArray) -> xr.DataArray:
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
    return xarray_ufunc(array.specific_humidity_from_mixing_ratio, w).assign_attrs({
        "standard_name": "specific_humidity",
        "units": "kg kg-1",
    })


def relative_humidity_from_dewpoint(t: xr.DataArray, td: xr.DataArray) -> xr.DataArray:
    r"""Compute the relative humidity from dew point temperature

    Parameters
    ----------
    t : xarray.DataArray
        Temperature (K)
    td: xarray.DataArray
        Dewpoint (K)


    Returns
    -------
    xarray.DataArray
        Relative humidity (%)


    The computation is based on the following formula:

    .. math::

        r = 100 \frac {e_{wsat}(td)}{e_{wsat}(t)}

    where :math:`e_{wsat}` is the :func:`saturation_vapour_pressure` over water.

    """
    return xarray_ufunc(array.relative_humidity_from_dewpoint, t, td).assign_attrs({
        "standard_name": "relative_humidity",
        "units": "%",
    })


def relative_humidity_from_specific_humidity(
    t: xr.DataArray, q: xr.DataArray, p: xr.DataArray
) -> xr.DataArray:
    r"""Compute the relative humidity from specific humidity.

    Parameters
    ----------
    t: xarray.DataArray
        Temperature (K)
    q: xarray.DataArray
        Specific humidity (kg/kg)
    p: xarray.DataArray
        Pressure (Pa)

    Returns
    -------
    xarray.DataArray
        Relative humidity (%)


    The computation is based on the following formula:

    .. math::

        r = 100 \frac {e(q, p)}{e_{msat}(t)}

    where:

        * :math:`e` is the vapour pressure (see :func:`vapour_pressure_from_specific_humidity`)
        * :math:`e_{msat}` is the :func:`saturation_vapour_pressure` based on the "mixed" phase

    """
    ...
    return xarray_ufunc(array.relative_humidity_from_specific_humidity, t, q, p).assign_attrs({
        "standard_name": "relative_humidity",
        "units": "%",
    })
