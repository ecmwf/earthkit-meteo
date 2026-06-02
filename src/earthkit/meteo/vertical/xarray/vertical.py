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
