# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,  # noqa: F401
    Iterable,
    TypeAlias,
    overload,
)

from earthkit.meteo.utils.decorators import dispatch

if TYPE_CHECKING:
    import xarray  # type: ignore[import]
    from earthkit.data import Field, FieldList  # type: ignore[import]

ArrayLike: TypeAlias = Any


@overload
def speed(u: "ArrayLike", v: "ArrayLike") -> "ArrayLike": ...


@overload
def speed(u: "xarray.DataArray", v: "xarray.DataArray") -> "xarray.DataArray": ...


@overload
def speed(u: "FieldList", v: "FieldList") -> "FieldList": ...


@overload
def speed(u: "Field", v: "Field") -> "Field": ...


def speed(
    u: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    v: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
) -> "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field":
    r"""Compute the wind speed/vector magnitude.

    Parameters
    ----------
    u: array-like | xarray.DataArray | FieldList | Field
        u wind/x vector component
    v: array-like | xarray.DataArray | FieldList | Field
        v wind/y vector component (same units as ``u``)

    Returns
    -------
    array-like | xarray.DataArray | FieldList | Field
        Wind speed/magnitude (same units as ``u`` and ``v``)


    Implementations
    ------------------------
    :func:`speed` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.array.speed` for array-like
    - :py:meth:`earthkit.meteo.wind.xarray.speed` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.fieldlist.speed` for FieldList | Field

    The function returns an object of the same type as the input arguments.

    """
    dispatched = dispatch(speed, array=True, fieldlist=True)
    return dispatched(u, v)


@overload
def direction(
    u: "ArrayLike",
    v: "ArrayLike",
    convention: str = "meteo",
    to_positive: bool = True,
) -> "ArrayLike": ...


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


@overload
def direction(
    u: "Field",
    v: "Field",
    convention: str = "meteo",
    to_positive: bool = True,
) -> "Field": ...


def direction(
    u: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    v: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    convention: str = "meteo",
    to_positive: bool = True,
) -> "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field":
    r"""Compute the direction/angle of a vector quantity.

    Parameters
    ----------
    u: array-like | xarray.DataArray | FieldList | Field
        u wind/x vector component
    v: array-like | xarray.DataArray | FieldList | Field
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
    array-like | xarray.DataArray | FieldList | Field
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

    - :py:meth:`earthkit.meteo.wind.array.direction` for array-like
    - :py:meth:`earthkit.meteo.wind.xarray.direction` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.fieldlist.direction` for FieldList | Field

    The function returns an object of the same type as the input arguments.

    """
    dispatched = dispatch(direction, array=True)
    return dispatched(u, v, convention=convention, to_positive=to_positive)


@overload
def xy_to_polar(x: "ArrayLike", y: "ArrayLike", convention: str = "meteo") -> tuple["ArrayLike", "ArrayLike"]: ...


@overload
def xy_to_polar(
    x: "xarray.DataArray", y: "xarray.DataArray", convention: str = "meteo"
) -> tuple["xarray.DataArray", "xarray.DataArray"]: ...


@overload
def xy_to_polar(x: "FieldList", y: "FieldList", convention: str = "meteo") -> tuple["FieldList", "FieldList"]: ...


@overload
def xy_to_polar(x: "Field", y: "Field", convention: str = "meteo") -> tuple["Field", "Field"]: ...


def xy_to_polar(
    x: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    y: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    convention: str = "meteo",
) -> tuple[
    "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
]:
    r"""Convert wind/vector data from xy representation to polar representation.

    Parameters
    ----------
    x: array-like | xarray.DataArray | FieldList | Field
        u wind/x vector component
    y: array-like | xarray.DataArray | FieldList | Field
        v wind/y vector component (same units as ``u``)
    convention: str
        Specify how the direction/angle component of the target polar coordinate
        system is interpreted. The possible values are as follows:

        * "meteo": the direction is the meteorological wind direction (see :func:`direction` for explanation)
        * "polar": the direction is measured anti-clockwise from the x axis (East/right) to the vector


    Returns
    -------
    array-like | xarray.DataArray | FieldList | Field
        Magnitude (same units as ``u``)
    array-like | xarray.DataArray | FieldList | Field
        Direction (degrees)

    Notes
    -----
    In the target xy representation the x axis points East while the y axis points North.


    Implementations
    ------------------------
    :func:`xy_to_polar` calls one of the following implementations depending on
    the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.array.xy_to_polar` for array-like
    - :py:meth:`earthkit.meteo.wind.xarray.xy_to_polar` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.fieldlist.xy_to_polar` for FieldList | Field

    The function returns an object of the same type as the input arguments.


    """
    dispatched = dispatch(xy_to_polar, array=True)
    return dispatched(x, y, convention=convention)


@overload
def polar_to_xy(
    magnitude: "ArrayLike",
    direction: "ArrayLike",
    convention: str = "meteo",
) -> tuple["ArrayLike", "ArrayLike"]: ...


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


@overload
def polar_to_xy(
    magnitude: "Field",
    direction: "Field",
    convention: str = "meteo",
) -> tuple["Field", "Field"]: ...


def polar_to_xy(
    magnitude: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    direction: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    convention: str = "meteo",
) -> tuple[
    "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
]:
    r"""Convert wind/vector data from polar representation to xy representation.

    Parameters
    ----------
    magnitude: array-like | xarray.DataArray | FieldList | Field
        Speed/magnitude of the vector
    direction: array-like | xarray.DataArray | FieldList | Field
        Direction of the vector (degrees)
    convention: str
        Specify how ``direction`` is interpreted. The possible values are as follows:

        * "meteo": ``direction`` is the meteorological wind direction
          (see :func:`direction` for explanation)
        * "polar": ``direction`` is the angle measured anti-clockwise from the x axis
          (East/right) to the vector

    Returns
    -------
    array-like | xarray.DataArray | FieldList | Field
        X vector component (same units as ``magnitude``)
    array-like | xarray.DataArray | FieldList | Field
        Y vector component (same units as ``magnitude``)


    Notes
    -----
    In the target xy representation the x axis points East while the y axis points North.


    Implementations
    ------------------------
    :func:`polar_to_xy` calls one of the following implementations depending on
    the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.array.polar_to_xy` for array-like
    - :py:meth:`earthkit.meteo.wind.xarray.polar_to_xy` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.fieldlist.polar_to_xy` for FieldList | Field

    The function returns an object of the same type as the input arguments.

    """
    dispatched = dispatch(polar_to_xy, array=True)
    return dispatched(magnitude, direction, convention=convention)


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


@overload
def w_from_omega(
    omega: "FieldList",
    t: "FieldList",
    p: "FieldList",
) -> "FieldList": ...


@overload
def w_from_omega(
    omega: "Field",
    t: "Field",
    p: "Field",
) -> "Field": ...


def w_from_omega(
    omega: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    t: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
    p: "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field",
) -> "ArrayLike" | "xarray.DataArray" | "FieldList" | "Field":
    r"""Compute the hydrostatic vertical velocity from pressure velocity.

    Parameters
    ----------
    omega : array-like | xarray.DataArray | FieldList | Field
        Hydrostatic pressure velocity (Pa/s)
    t : array-like | xarray.DataArray | FieldList | Field
        Temperature (K)
    p : array-like | xarray.DataArray | FieldList | Field
        Pressure (Pa)

    Returns
    -------
    array-like | xarray.DataArray | FieldList | Field
        Hydrostatic vertical velocity (m/s)

    Notes
    -----
    The computation is based on the following hydrostatic formula:

    .. math::

        w = - \frac{\omega  t R_{d}}{p g}

    where

        * :math:`R_{d}` is the specific gas constant for dry air (see :data:`earthkit.meteo.constants.Rd`).
        * :math:`g` is the gravitational acceleration (see :data:`earthkit.meteo.constants.g`)


    Implementations
    ------------------------
    :func:`w_from_omega` calls one of the following implementations depending on
    the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.array.w_from_omega` for array-like
    - :py:meth:`earthkit.meteo.wind.xarray.w_from_omega` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.fieldlist.w_from_omega` for FieldList | Field

    The function returns an object of the same type as the input arguments.

    """
    dispatched = dispatch(w_from_omega, array=True)
    return dispatched(omega, t, p)


@overload
def coriolis(lat: "ArrayLike") -> "ArrayLike": ...


@overload
def coriolis(lat: "xarray.DataArray") -> "xarray.DataArray": ...


@overload
def coriolis(lat: "FieldList") -> "FieldList": ...


@overload
def coriolis(lat: "Field") -> "Field": ...


def coriolis(
    lat: "xarray.DataArray" | "ArrayLike" | "FieldList" | "Field",
) -> "xarray.DataArray" | "ArrayLike" | "FieldList" | "Field":
    r"""Compute the Coriolis parameter.

    Parameters
    ----------
    lat : array-like | xarray.DataArray | FieldList | Field
        Latitude (degrees)

    Returns
    -------
    array-like | xarray.DataArray | FieldList | Field
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
    :func:`coriolis` calls one of the following implementations depending on
    the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.array.coriolis` for array-like
    - :py:meth:`earthkit.meteo.wind.xarray.coriolis` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.fieldlist.coriolis` for FieldList | Field

    The function returns an object of the same type as the input arguments.

    """
    dispatched = dispatch(coriolis, array=True, fieldlist=True)
    return dispatched(lat)


@overload
def windrose(
    speed: "ArrayLike",
    direction: "ArrayLike",
    sectors: int = 16,
    speed_bins: Iterable[float] | None = None,
    percent: bool = True,
) -> tuple["ArrayLike", "ArrayLike"]: ...


@overload
def windrose(
    speed: "xarray.DataArray",
    direction: "xarray.DataArray",
    sectors: int = 16,
    speed_bins: Iterable[float] | None = None,
    percent: bool = True,
) -> tuple["xarray.DataArray", "xarray.DataArray"]: ...


def windrose(
    speed: "ArrayLike" | "xarray.DataArray",
    direction: "ArrayLike" | "xarray.DataArray",
    sectors: int = 16,
    speed_bins: Iterable[float] | None = None,
    percent: bool = True,
) -> tuple[
    "ArrayLike" | "xarray.DataArray",
    "ArrayLike" | "xarray.DataArray",
]:
    """Generate windrose data.

    Parameters
    ----------
    speed : array-like | xarray.DataArray
        Speed.
    direction : array-like | xarray.DataArray
        Meteorological wind direction (degrees). Values must be in [0, 360].
    sectors : int, optional
        Number of sectors the 360 degree range is split into.
    speed_bins : Iterable[float] | None, optional
        Speed bins. Must contain at least two values.
    percent : bool, optional
        If True, return percentages. If False, return counts.

    Returns
    -------
    array-like | xarray.DataArray
        2D histogram over speed and direction bins.
    array-like | xarray.DataArray
        Direction bin edges (degrees).


    Implementations
    ------------------------
    :func:`windrose` calls one of the following implementations depending on
    the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.array.windrose` for array-like
    - :py:meth:`earthkit.meteo.wind.xarray.windrose` for xarray.DataArray

    The function returns an object of the same type as the input arguments.

    """
    dispatched = dispatch(windrose, fieldlist=False, array=True)
    return dispatched(speed, direction, sectors=sectors, speed_bins=speed_bins, percent=percent)
