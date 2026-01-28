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

from earthkit.meteo.utils.meteo_decorator import dispatch

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
def speed(u: Any, v: Any) -> Any:
    r"""Compute the wind speed/vector magnitude.

    Parameters
    ----------
    u: Any
        u wind/x vector component
    v: Any
        v wind/y vector component (same units as ``u``)

    Returns
    -------
    Any
        Wind speed/magnitude (same units as ``u`` and ``v``)

    """
    pass


@overload
def direction(
    u: "xarray.DataArray", v: "xarray.DataArray", convention: str = "meteo", to_positive: bool = True
) -> "xarray.DataArray":
    r"""Compute the direction/angle of a vector quantity."""
    ...


@dispatch
def direction(u: Any, v: Any, convention: str = "meteo", to_positive: bool = True) -> Any:
    r"""Compute the direction/angle of a vector quantity."""
    pass


@overload
def xy_to_polar(
    x: "xarray.DataArray",
    y: "xarray.DataArray",
    convention: str = "meteo",
) -> tuple["xarray.DataArray", "xarray.DataArray"]:
    r"""Convert wind/vector data from xy representation to polar representation."""
    ...


@dispatch
def xy_to_polar(x: Any, y: Any, convention: str = "meteo") -> tuple[Any, Any]:
    r"""Convert wind/vector data from xy representation to polar representation."""
    pass


@overload
def polar_to_xy(
    magnitude: "xarray.DataArray",
    direction: "xarray.DataArray",
    convention: str = "meteo",
) -> tuple["xarray.DataArray", "xarray.DataArray"]:
    r"""Convert wind/vector data from polar representation to xy representation."""
    ...


@dispatch
def polar_to_xy(magnitude: Any, direction: Any, convention: str = "meteo") -> tuple[Any, Any]:
    r"""Convert wind/vector data from polar representation to xy representation."""
    pass


@overload
def w_from_omega(
    omega: "xarray.DataArray",
    t: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the hydrostatic vertical velocity from pressure velocity"""
    ...


@dispatch
def w_from_omega(omega: Any, t: Any, p: Any) -> Any:
    r"""Compute the hydrostatic vertical velocity from pressure velocity"""
    pass


@overload
def coriolis(lat: "xarray.DataArray") -> "xarray.DataArray":
    r"""Compute the Coriolis parameter"""
    ...


@dispatch
def coriolis(lat: Any) -> Any:
    r"""Compute the Coriolis parameter"""
    pass


@overload
def windrose(
    speed: "xarray.DataArray",
    direction: "xarray.DataArray",
    sectors: int = 16,
    speed_bins: Iterable[float] | None = None,
    percent: bool = True,
) -> tuple["xarray.DataArray", "xarray.DataArray"]:
    r"""Generate windrose data"""
    ...


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
