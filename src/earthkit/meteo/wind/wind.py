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
    from earthkit.data import FieldList  # type: ignore[import]

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
def speed(u: "FieldList", v: "FieldList") -> "FieldList":
    r"""Compute the wind speed/vector magnitude.

    Parameters
    ----------
    u: "FieldList"
        u wind/x vector component
    v: "FieldList"
        v wind/y vector component (same units as ``u``)

    Returns
    -------
    "FieldList"
        Wind speed/magnitude (same units as ``u`` and ``v``)

    """
    ...


def speed(
    u: "xarray.DataArray" | "FieldList", v: "xarray.DataArray" | "FieldList"
) -> "xarray.DataArray" | "FieldList":
    r"""Compute the wind speed/vector magnitude.

    Parameters
    ----------
    u: xarray.DataArray, FieldList
        u wind/x vector component
    v: xarray.DataArray, FieldList
        v wind/y vector component (same units as ``u``)

    Returns
    -------
    xarray.DataArray, FieldList
        Wind speed/magnitude (same units as ``u`` and ``v``)


    Implementations
    ------------------------
    :func:`speed` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.xarray.speed` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.fieldlist.speed` for FieldList

    """
    return dispatch(speed, u, v)


@overload
def direction(
    u: "xarray.DataArray",
    v: "xarray.DataArray",
    convention: str = "meteo",
    to_positive: bool = True,
) -> "xarray.DataArray": ...


@overload
def direction(
    u: "FieldList",
    v: "FieldList",
    convention: str = "meteo",
    to_positive: bool = True,
) -> "FieldList": ...


def direction(
    u: "xarray.DataArray" | "FieldList",
    v: "xarray.DataArray" | "FieldList",
    convention: str = "meteo",
    to_positive: bool = True,
) -> "xarray.DataArray" | "FieldList":
    r"""Compute the direction/angle of a vector quantity.

    Parameters
    ----------
    u: xarray.DataArray | FieldList
        u wind/x vector component
    v: xarray.DataArray | FieldList
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
    xarray.DataArray | FieldList
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
    - :py:meth:`earthkit.meteo.wind.fieldlist.direction` for FieldList

    """
    return dispatch(direction, u, v, convention=convention, to_positive=to_positive)


@overload
def xy_to_polar(
    x: "xarray.DataArray", y: "xarray.DataArray", convention: str = "meteo"
) -> tuple["xarray.DataArray", "xarray.DataArray"]: ...


@overload
def xy_to_polar(
    x: "FieldList", y: "FieldList", convention: str = "meteo"
) -> tuple["FieldList", "FieldList"]: ...


def xy_to_polar(
    x: "xarray.DataArray" | "FieldList", y: "xarray.DataArray" | "FieldList", convention: str = "meteo"
) -> tuple["xarray.DataArray" | "FieldList", "xarray.DataArray" | "FieldList"]:
    r"""Convert wind/vector data from xy representation to polar representation.

    Parameters
    ----------
    x: "xarray.DataArray" | "FieldList"
        u wind/x vector component
    y: "xarray.DataArray" | "FieldList"
        v wind/y vector component (same units as ``u``)
    convention: str
        Specify how the direction/angle component of the target polar coordinate
        system is interpreted. The possible values are as follows:

        * "meteo": the direction is the meteorological wind direction (see :func:`direction` for explanation)
        * "polar": the direction is measured anti-clockwise from the x axis (East/right) to the vector


    Returns
    -------
    "xarray.DataArray" | "FieldList"
        Magnitude (same units as ``u``)
    "xarray.DataArray" | "FieldList"
        Direction (degrees)

    Notes
    -----
    In the target xy representation the x axis points East while the y axis points North.


    Implementations
    ------------------------
    :func:`xy_to_polar` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.xarray.xy_to_polar` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.fieldlist.xy_to_polar` for FieldList


    """
    return dispatch(xy_to_polar, x, y, convention=convention)


@overload
def polar_to_xy(
    magnitude: "xarray.DataArray",
    direction: "xarray.DataArray",
    convention: str = "meteo",
) -> tuple["xarray.DataArray", "xarray.DataArray"]: ...


@overload
def polar_to_xy(
    magnitude: "FieldList",
    direction: "FieldList",
    convention: str = "meteo",
) -> tuple["FieldList", "FieldList"]: ...


def polar_to_xy(
    magnitude: "xarray.DataArray" | "FieldList",
    direction: "xarray.DataArray" | "FieldList",
    convention: str = "meteo",
) -> tuple["xarray.DataArray" | "FieldList", "xarray.DataArray" | "FieldList" :]:
    r"""Convert wind/vector data from polar representation to xy representation.

    Parameters
    ----------
    magnitude: "xarray.DataArray" | "FieldList"
        Speed/magnitude of the vector
    direction: "xarray.DataArray" | "FieldList"
        Direction of the vector (degrees)
    convention: str
        Specify how ``direction`` is interpreted. The possible values are as follows:

        * "meteo": ``direction`` is the meteorological wind direction
          (see :func:`direction` for explanation)
        * "polar": ``direction`` is the angle measured anti-clockwise from the x axis
          (East/right) to the vector

    Returns
    -------
    "xarray.DataArray" | "FieldList"
        X vector component (same units as ``magnitude``)
    "xarray.DataArray" | "FieldList"
        Y vector component (same units as ``magnitude``)


    Notes
    -----
    In the target xy representation the x axis points East while the y axis points North.


    Implementations
    ------------------------
    :func:`polar_to_xy` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.xarray.polar_to_xy` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.fieldlist.polar_to_xy` for FieldList

    """
    return dispatch(polar_to_xy, magnitude, direction, convention=convention)


@overload
def w_from_omega(
    omega: "xarray.DataArray",
    t: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray": ...


@overload
def w_from_omega(
    omega: "FieldList",
    t: "FieldList",
    p: "FieldList" | "ArrayLike" | None,
) -> "FieldList": ...


def w_from_omega(
    omega: "xarray.DataArray" | "FieldList",
    t: "xarray.DataArray" | "FieldList",
    p: "xarray.DataArray" | "FieldList" | "ArrayLike" | None,
) -> "xarray.DataArray" | "FieldList":
    r"""Compute the hydrostatic vertical velocity from pressure velocity

    Parameters
    ----------
    omega : array-like
        Hydrostatic pressure velocity (Pa/s)
    t : array-like
        Temperature (K)
    p : "xarray.DataArray" | "FieldList"  | ArrayLike | None
        Pressure (Pa). When ``omega`` and ``t`` are FieldList, ``p`` can be one of the following:

        - FieldList with the same number of fields as ``omega``
        - array-like with the same number of elements as the number of fields in ``omega``
        - None. In this case the pressure is taken from the level information of each field
        in ``omega`` (only isobaric levels are supported).

    Returns
    -------
    "xarray.DataArray" | "FieldList"
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
    - :py:meth:`earthkit.meteo.wind.fieldlist.w_from_omega` for FieldList

    """
    return dispatch(w_from_omega, omega, t, p)


@overload
def coriolis(lat: "xarray.DataArray") -> "xarray.DataArray": ...


@overload
def coriolis(lat: "FieldList") -> "FieldList": ...


def coriolis(lat: "xarray.DataArray" | "FieldList") -> "xarray.DataArray" | "FieldList":
    r"""Compute the Coriolis parameter

    Parameters
    ----------
    lat : Xarray.DataArray | FieldList
        Latitude (degrees)

    Returns
    -------
    Xarray.DataArray | FieldList
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
    - :py:meth:`earthkit.meteo.wind.fieldlist.coriolis` for FieldList

    """
    return dispatch(coriolis, lat)
