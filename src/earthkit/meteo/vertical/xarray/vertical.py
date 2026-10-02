# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import xarray as xr

from earthkit.meteo import constants
from earthkit.meteo.utils.decorators import xarray_ufunc

from .. import array


def geopotential_height_from_geopotential(z: xr.DataArray) -> xr.DataArray:
    r"""Compute geopotential height from geopotential.

    Parameters
    ----------
    z : xr.DataArray
        Geopotential (m2/s2)

    Returns
    -------
    xr.DataArray
        Geopotential height (m)


    The computation is based on the following definition:

    .. math::

        gh = \frac{z}{g}

    where :math:`g` is the gravitational acceleration on the surface of
    the Earth (see :py:attr:`earthkit.meteo.constants.g`)
    """
    return xarray_ufunc(array.geopotential_height_from_geopotential, z).assign_attrs({
        "standard_name": "geopotential_height",
        "units": "m",
    })


def geopotential_from_geopotential_height(gh: xr.DataArray) -> xr.DataArray:
    r"""Compute geopotential height from geopotential.

    Parameters
    ----------
    gh : xr.DataArray
        Geopotential height (m)

    Returns
    -------
    xr.DataArray
        Geopotential height (m)


    The computation is based on the following definition:

    .. math::

        z = gh  g

    where :math:`g` is the gravitational acceleration on the surface of
    the Earth (see :py:attr:`earthkit.meteo.constants.g`)
    """
    return xarray_ufunc(array.geopotential_from_geopotential_height, gh).assign_attrs({
        "standard_name": "geopotential",
        "units": "m2 s-2",
    })


def geopotential_height_from_geometric_height(h: xr.DataArray, R_earth: float = constants.R_earth) -> xr.DataArray:
    r"""Compute the geopotential height from geometric height.

    Parameters
    ----------
    h : xr.DataArray
        Geometric height with respect to the sea level (m)
    R_earth : float
        Average radius of the Earth (m)

    Returns
    -------
    xr.DataArray
        Geopotential height (m)


    The computation is based on the following formula:

    .. math::

        gh = \frac{h  R_{earth}}{R_{earth} + h}

    where :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`earthkit.meteo.constants.R_earth`)
    """
    return xarray_ufunc(array.geopotential_height_from_geometric_height, h, R_earth).assign_attrs({
        "standard_name": "geopotential_height",
        "units": "m",
    })


def geopotential_from_geometric_height(h: xr.DataArray, R_earth: float = constants.R_earth) -> xr.DataArray:
    r"""Compute the geopotential from geometric height.

    Parameters
    ----------
    h : xr.DataArray
        Geometric height with respect to the sea level (m)
    R_earth : float
        Average radius of the Earth (m)

    Returns
    -------
    xr.DataArray
        Geopotential (m2/s2)


    The computation is based on the following formula:

    .. math::

        z = \frac{h  g  R_{earth}}{R_{earth} + h}

    where

        * :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`earthkit.meteo.constants.R_earth`)
        * :math:`g` is the gravitational acceleration on the surface of
          the Earth (see :py:attr:`earthkit.meteo.constants.g`)
    """
    return xarray_ufunc(array.geopotential_from_geometric_height, h, R_earth).assign_attrs({
        "standard_name": "geopotential",
        "units": "m2 s-2",
    })


def geometric_height_from_geopotential_height(gh: xr.DataArray, R_earth: float = constants.R_earth) -> xr.DataArray:
    r"""Compute the geometric height from geopotential height.

    Parameters
    ----------
    gh : xr.DataArray
        Geopotential height (m)
    R_earth : float
        Average radius of the Earth (m)

    Returns
    -------
    xr.DataArray
        Geometric height (m)


    The computation is based on the following formula:

    .. math::

        h = \frac{R_{earth}  gh}{R_{earth} - gh}

    where :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`earthkit.meteo.constants.R_earth`)
    """
    return xarray_ufunc(array.geometric_height_from_geopotential_height, gh, R_earth).assign_attrs({
        "standard_name": "geometric_height",
        "units": "m",
    })


def geometric_height_from_geopotential(z: xr.DataArray, R_earth: float = constants.R_earth) -> xr.DataArray:
    r"""Compute the geometric height from geopotential.

    Parameters
    ----------
    z : xr.DataArray
        Geopotential (m2/s2)
    R_earth : float
        Average radius of the Earth (m)

    Returns
    -------
    xr.DataArray
        Geometric height (m)


    The computation is based on the following formula:

    .. math::

        h = \frac{R_{earth} \frac{z}{g}}{R_{earth} - \frac{z}{g}}

    where

        * :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`earthkit.meteo.constants.R_earth`)
        * :math:`g` is the gravitational acceleration on the surface of
          the Earth (see :py:attr:`earthkit.meteo.constants.g`)
    """
    return xarray_ufunc(array.geometric_height_from_geopotential, z, R_earth).assign_attrs({
        "standard_name": "geometric_height",
        "units": "m",
    })


def extrapolate_temperature_below_surface(
    t_sfc: xr.DataArray, h_sfc: xr.DataArray, p_sfc: xr.DataArray, target_p: xr.DataArray | float
) -> xr.DataArray:
    r"""Extrapolate temperature from the surface to pressure levels below the surface.

    Parameters
    ----------
    t_sfc : xr.DataArray
        Surface temperature (K)
    h_sfc : xr.DataArray
        Surface height above sea level (m)
    p_sfc : xr.DataArray
        Surface pressure (Pa)
    target_p : xr.DataArray | float
        Target pressure (Pa). Broadcast against the other inputs following xarray rules,
        e.g. ``xr.DataArray([85000.0, 100000.0], dims="z")`` adds a ``z`` dimension.

    Returns
    -------
    xr.DataArray
        Temperature (K)

    See :func:`earthkit.meteo.vertical.array.extrapolate_temperature_below_surface`
    for the algorithm.
    """
    return xarray_ufunc(array.extrapolate_temperature_below_surface, t_sfc, h_sfc, p_sfc, target_p).assign_attrs({
        "standard_name": "air_temperature",
        "units": "K",
    })


def extrapolate_geopotential_below_surface(
    t_sfc: xr.DataArray, h_sfc: xr.DataArray, p_sfc: xr.DataArray, target_p: xr.DataArray | float
) -> xr.DataArray:
    r"""Extrapolate geopotential from the surface to pressure levels below the surface.

    Parameters
    ----------
    t_sfc : xr.DataArray
        Surface temperature (K)
    h_sfc : xr.DataArray
        Surface height above sea level (m)
    p_sfc : xr.DataArray
        Surface pressure (Pa)
    target_p : xr.DataArray | float
        Target pressure (Pa). Broadcast against the other inputs following xarray rules,
        e.g. ``xr.DataArray([85000.0, 100000.0], dims="z")`` adds a ``z`` dimension.

    Returns
    -------
    xr.DataArray
        Geopotential (m2/s2)

    See :func:`earthkit.meteo.vertical.array.extrapolate_geopotential_below_surface`
    for the algorithm.
    """
    return xarray_ufunc(array.extrapolate_geopotential_below_surface, t_sfc, h_sfc, p_sfc, target_p).assign_attrs({
        "standard_name": "geopotential",
        "units": "m2 s-2",
    })
