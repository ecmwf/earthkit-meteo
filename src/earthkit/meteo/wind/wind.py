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

if TYPE_CHECKING:
    import xarray  # type: ignore[import]
    from earthkit.data import FieldList  # type: ignore[import]


# TODO: move these underscore functions to meteo.utils
def _is_xarray(obj: Any) -> bool:
    from earthkit.meteo.utils import is_module_loaded

    if not is_module_loaded("xarray"):
        return False

    try:
        import xarray as xr

        return isinstance(obj, (xr.DataArray, xr.Dataset))
    except (ImportError, RuntimeError, SyntaxError):
        return False


def _is_fieldlist(obj: Any) -> bool:
    from earthkit.meteo.utils import is_module_loaded

    if not is_module_loaded("earthkit.data"):
        return False

    try:
        from earthkit.data import FieldList

        return isinstance(obj, FieldList)
    except ImportError:
        return False


def _call(func: str, *args: Any, **kwargs: Any) -> Any:
    if _is_xarray(args[0]):
        from . import xarray as _module
    elif _is_fieldlist(args[0]):
        from . import fieldlist as _module

    return getattr(_module, func)(*args, **kwargs)


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
    return _call("speed", u, v)


@overload
def direction(
    u: "xarray.DataArray", v: "xarray.DataArray", convention: str = "meteo", to_positive: bool = True
) -> "xarray.DataArray":
    r"""Compute the direction/angle of a vector quantity."""
    ...


def direction(u: Any, v: Any, convention: str = "meteo", to_positive: bool = True) -> Any:
    r"""Compute the direction/angle of a vector quantity."""

    return _call("direction", u, v, convention=convention, to_positive=to_positive)


@overload
def xy_to_polar(
    x: "xarray.DataArray",
    y: "xarray.DataArray",
    convention: str = "meteo",
) -> tuple["xarray.DataArray", "xarray.DataArray"]:
    r"""Convert wind/vector data from xy representation to polar representation."""
    ...


def xy_to_polar(x: Any, y: Any, convention: str = "meteo") -> tuple[Any, Any]:
    r"""Convert wind/vector data from xy representation to polar representation."""

    return _call("xy_to_polar", x, y, convention=convention)


@overload
def polar_to_xy(
    magnitude: "xarray.DataArray",
    direction: "xarray.DataArray",
    convention: str = "meteo",
) -> tuple["xarray.DataArray", "xarray.DataArray"]:
    r"""Convert wind/vector data from polar representation to xy representation."""
    ...


def polar_to_xy(magnitude: Any, direction: Any, convention: str = "meteo") -> tuple[Any, Any]:
    r"""Convert wind/vector data from polar representation to xy representation."""
    return _call("polar_to_xy", magnitude, direction, convention=convention)


@overload
def w_from_omega(
    omega: "xarray.DataArray",
    t: "xarray.DataArray",
    p: "xarray.DataArray",
) -> "xarray.DataArray":
    r"""Compute the hydrostatic vertical velocity from pressure velocity"""
    ...


def w_from_omega(omega: Any, t: Any, p: Any) -> Any:
    r"""Compute the hydrostatic vertical velocity from pressure velocity"""
    return _call("w_from_omega", omega, t, p)


@overload
def coriolis(lat: "xarray.DataArray") -> "xarray.DataArray":
    r"""Compute the Coriolis parameter"""
    ...


def coriolis(lat: Any) -> Any:
    r"""Compute the Coriolis parameter"""

    return _call("coriolis", lat)


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


def windrose(
    speed: Any,
    direction: Any,
    sectors: int = 16,
    speed_bins: Iterable[float] | None = None,
    percent: bool = True,
) -> tuple[Any, Any]:
    r"""Generate windrose data"""

    return _call(
        "windrose",
        speed,
        direction,
        sectors=sectors,
        speed_bins=speed_bins,
        percent=percent,
    )
