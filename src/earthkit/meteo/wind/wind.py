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
from typing import TypeAlias
from typing import overload

from earthkit.meteo.utils.decorators import dispatch

if TYPE_CHECKING:
    import xarray  # type: ignore[import]

ArrayLike: TypeAlias = Any


@overload
def speed(u: "xarray.DataArray", v: "xarray.DataArray") -> "xarray.DataArray":
    r"""Compute the wind speed/vector magnitude.

    Parameters
    ----------
    u: xarray.DataArray
        u wind/x vector component
    v: xarray.DataArray
        v wind/y vector component (same units as ``u``)

    Returns
    -------
    xarray.DataArray
        Wind speed/magnitude (same units as ``u`` and ``v``)

    """
    ...


@overload
def speed(u: "ArrayLike", v: "ArrayLike") -> "ArrayLike":
    r"""Compute the wind speed/vector magnitude.

    Parameters
    ----------
    u: array-like
        u wind/x vector component
    v: array-like
        v wind/y vector component (same units as ``u``)

    Returns
    -------
    array-like
        Wind speed/magnitude (same units as ``u`` and ``v``)

    """
    ...


def speed(
    u: "xarray.DataArray" | "ArrayLike",
    v: "xarray.DataArray" | "ArrayLike",
) -> "xarray.DataArray" | "ArrayLike":
    r"""Compute the wind speed/vector magnitude.

    Parameters
    ----------
    u: xarray.DataArray, array-like
        u wind/x vector component
    v: xarray.DataArray, array-like
        v wind/y vector component (same units as ``u``)

    Returns
    -------
    xarray.DataArray, array-like
        Wind speed/magnitude (same units as ``u`` and ``v``)


    Implementations
    ------------------------
    :func:`speed` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.xarray.speed` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.array.speed` for array-like

    The function returns an object of the same type as the input arguments.

    """
    return dispatch(speed, fieldlist=False, array=True)(u, v)


@overload
def direction(
    u: "xarray.DataArray",
    v: "xarray.DataArray",
    convention: str = "meteo",
    to_positive: bool = True,
) -> "xarray.DataArray": ...


@overload
def direction(
    u: "ArrayLike",
    v: "ArrayLike",
    convention: str = "meteo",
    to_positive: bool = True,
) -> "ArrayLike": ...


def direction(
    u: "xarray.DataArray" | "ArrayLike",
    v: "xarray.DataArray" | "ArrayLike",
    convention: str = "meteo",
    to_positive: bool = True,
) -> "xarray.DataArray" | "ArrayLike":
    r"""Compute the direction/angle of a vector quantity.

    Parameters
    ----------
    u: xarray.DataArray | array-like
        u wind/x vector component
    v: xarray.DataArray | array-like
        v wind/y vector component (same units as ``u``)
    convention: str, optional
        Specify how the direction/angle is interpreted. The possible values are as follows:

        * "meteo": the direction is the meteorological wind direction (see below for explanation)
        * "polar": the direction is measured anti-clockwise from the x axis (East/right) to the vector

    to_positive: bool, optional
        If True, the resulting values are mapped into the [0, 360] range when
        ``convention`` is "polar". Otherwise they lie in the [-180, 180] range.


    Returns
    -------
    xarray.DataArray | array-like
        Direction/angle (degrees)


    Notes
    -----
    The meteorological wind direction is the direction from which the wind is
    blowing. Wind direction increases clockwise such that a northerly wind
    is 0°, an easterly wind is 90°, a southerly wind is 180°, and a westerly
    wind is 270°. The figure below illustrates how it is related to the actual
    orientation of the wind vector:

    .. image:: /_static/wind_direction.png
        :width: 400px


    Implementations
    ------------------------
    :func:`direction` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.xarray.direction` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.array.direction` for array-like

    The function returns an object of the same type as the input arguments.

    """
    return dispatch(direction, fieldlist=False, array=True)(
        u, v, convention=convention, to_positive=to_positive
    )


@overload
def xy_to_polar(
    x: "xarray.DataArray", y: "xarray.DataArray", convention: str = "meteo"
) -> tuple["xarray.DataArray", "xarray.DataArray"]: ...


@overload
def xy_to_polar(
    x: "ArrayLike", y: "ArrayLike", convention: str = "meteo"
) -> tuple["ArrayLike", "ArrayLike"]: ...


def xy_to_polar(
    x: "xarray.DataArray" | "ArrayLike",
    y: "xarray.DataArray" | "ArrayLike",
    convention: str = "meteo",
) -> tuple[
    "xarray.DataArray" | "ArrayLike",
    "xarray.DataArray" | "ArrayLike",
]:
    r"""Convert wind/vector data from xy representation to polar representation.

    Parameters
    ----------
    x: "xarray.DataArray" | array-like
        u wind/x vector component
    y: "xarray.DataArray" | array-like
        v wind/y vector component (same units as ``u``)
    convention: str
        Specify how the direction/angle component of the target polar coordinate
        system is interpreted. The possible values are as follows:

        * "meteo": the direction is the meteorological wind direction (see :func:`direction` for explanation)
        * "polar": the direction is measured anti-clockwise from the x axis (East/right) to the vector


    Returns
    -------
    "xarray.DataArray" | array-like
        Magnitude (same units as ``u``)
    "xarray.DataArray" | array-like
        Direction (degrees)

    Notes
    -----
    In the target xy representation the x axis points East while the y axis points North.


    Implementations
    ------------------------
    :func:`xy_to_polar` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.xarray.xy_to_polar` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.array.xy_to_polar` for array-like

    The function returns an object of the same type as the input arguments.


    """
    return dispatch(xy_to_polar, fieldlist=False, array=True)(x, y, convention=convention)


@overload
def polar_to_xy(
    magnitude: "xarray.DataArray",
    direction: "xarray.DataArray",
    convention: str = "meteo",
) -> tuple["xarray.DataArray", "xarray.DataArray"]: ...


@overload
def polar_to_xy(
    magnitude: "ArrayLike",
    direction: "ArrayLike",
    convention: str = "meteo",
) -> tuple["ArrayLike", "ArrayLike"]: ...


def polar_to_xy(
    magnitude: "xarray.DataArray" | "ArrayLike",
    direction: "xarray.DataArray" | "ArrayLike",
    convention: str = "meteo",
) -> tuple[
    "xarray.DataArray" | "ArrayLike",
    "xarray.DataArray" | "ArrayLike",
]:
    r"""Convert wind/vector data from polar representation to xy representation.

    Parameters
    ----------
    magnitude: "xarray.DataArray" | array-like
        Speed/magnitude of the vector
    direction: "xarray.DataArray" | array-like
        Direction of the vector (degrees)
    convention: str
        Specify how ``direction`` is interpreted. The possible values are as follows:

        * "meteo": ``direction`` is the meteorological wind direction
          (see :func:`direction` for explanation)
        * "polar": ``direction`` is the angle measured anti-clockwise from the x axis
          (East/right) to the vector

    Returns
    -------
    "xarray.DataArray" | array-like
        X vector component (same units as ``magnitude``)
    "xarray.DataArray" | array-like
        Y vector component (same units as ``magnitude``)


    Notes
    -----
    In the target xy representation the x axis points East while the y axis points North.


    Implementations
    ------------------------
    :func:`polar_to_xy` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.xarray.polar_to_xy` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.array.polar_to_xy` for array-like

    The function returns an object of the same type as the input arguments.

    """
    return dispatch(polar_to_xy, fieldlist=False, array=True)(magnitude, direction, convention=convention)


@overload
def w_from_omega(
    omega: "ArrayLike",
    t: "ArrayLike",
    p: "ArrayLike",
) -> "ArrayLike": ...


@overload
def w_from_omega(
    omega: "xarray.DataArray",
    t: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray": ...


def w_from_omega(
    omega: "xarray.DataArray" | "ArrayLike",
    t: "xarray.DataArray" | "ArrayLike",
    p: "xarray.DataArray" | "ArrayLike",
) -> "xarray.DataArray" | "ArrayLike":
    r"""Compute the hydrostatic vertical velocity from pressure velocity

    Parameters
    ----------
    omega : array-like
        Hydrostatic pressure velocity (Pa/s)
    t : array-like
        Temperature (K)
    p : "xarray.DataArray" | ArrayLike
        Pressure (Pa)

    Returns
    -------
    "xarray.DataArray" | array-like
        Hydrostatic vertical velocity (m/s)

    Notes
    -----
    The computation is based on the following hydrostatic formula:

    .. math::

        w = - \frac{\omega\; t R_{d}}{p g}

    where

        * :math:`R_{d}` is the specific gas constant for dry air (see :data:`earthkit.meteo.constants.Rd`).
        * :math:`g` is the gravitational acceleration (see :data:`earthkit.meteo.constants.g`)


    Implementations
    ------------------------
    :func:`w_from_omega` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.xarray.w_from_omega` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.array.w_from_omega` for array-like

    The function returns an object of the same type as the input arguments.

    """
    return dispatch(w_from_omega, fieldlist=False, array=True)(omega, t, p)


@overload
def coriolis(lat: "xarray.DataArray") -> "xarray.DataArray": ...


@overload
def coriolis(lat: "ArrayLike") -> "ArrayLike": ...


def coriolis(lat: "xarray.DataArray" | "ArrayLike") -> "xarray.DataArray" | "ArrayLike":
    r"""Compute the Coriolis parameter

    Parameters
    ----------
    lat : Xarray.DataArray | array-like
        Latitude (degrees)

    Returns
    -------
    Xarray.DataArray | array-like
        The Coriolis parameter (:math:`s^{-1}`)

    Notes
    -----
    The Coriolis parameter is defined by the following formula:

    .. math::

        f = 2 \Omega sin(\phi)

    where :math:`\Omega` is the rotation rate of Earth
    (see :data:`earthkit.meteo.constants.omega`) and :math:`\phi` is the latitude.

    Implementations
    ------------------------
    :func:`coriolis` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.xarray.coriolis` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.array.coriolis` for array-like

    The function returns an object of the same type as the input arguments.

    """
    return dispatch(coriolis, fieldlist=False, array=True)(lat)
