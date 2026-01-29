#
# # (C) Copyright 2021 ECMWF.
# #
# # This software is licensed under the terms of the Apache Licence Version 2.0
# # which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# # In applying this licence, ECMWF does not waive the privileges and immunities
# # granted to it by virtue of its status as an intergovernmental organisation
# # nor does it submit to any jurisdiction.
# ##
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any  # noqa: F401
from typing import Iterable
from typing import overload

from earthkit.meteo.utils.decorators import dispatch

if TYPE_CHECKING:
    import xarray  # type: ignore[import]
    from earthkit.data import FieldList  # type: ignore[import]


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


@dispatch
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
    pass


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


@dispatch
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
    :func:`speed` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.wind.xarray.direction` for xarray.DataArray
    - :py:meth:`earthkit.meteo.wind.fieldlist.direction` for FieldList

    """
    pass


@dispatch
def xy_to_polar(x: Any, y: Any, convention: str = "meteo") -> tuple[Any, Any]:
    r"""Convert wind/vector data from xy representation to polar representation."""
    pass


@dispatch
def polar_to_xy(magnitude: Any, direction: Any, convention: str = "meteo") -> tuple[Any, Any]:
    r"""Convert wind/vector data from polar representation to xy representation."""
    pass


@dispatch
def w_from_omega(omega: Any, t: Any, p: Any) -> Any:
    r"""Compute the hydrostatic vertical velocity from pressure velocity"""
    pass


@dispatch
def coriolis(lat: Any) -> Any:
    r"""Compute the Coriolis parameter"""
    pass


@dispatch
def windrose(
    speed: Any,
    direction: Any,
    sectors: int = 16,
    speed_bins: Iterable[float] | None = None,
    percent: bool = True,
) -> tuple[Any, Any]:
    r"""Generate windrose data"""
    pass
