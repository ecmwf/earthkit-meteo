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

from typing import Any, Iterable

from . import array


def _is_xarray(obj: Any) -> bool:
    try:
        import xarray as xr
    except (ImportError, RuntimeError, SyntaxError):
        return False
    return isinstance(obj, (xr.DataArray, xr.Dataset))


def speed(u: Any, v: Any) -> Any:
    r"""Compute the wind speed/vector magnitude.

    Parameters
    ----------
    u: array-like or xarray.DataArray
        u wind/x vector component
    v: array-like or xarray.DataArray
        v wind/y vector component (same units as ``u``)

    Returns
    -------
    array-like or xarray.DataArray
        Wind speed/magnitude (same units as ``u`` and ``v``)
    """
    if _is_xarray(u):
        from . import xarray as xarray_module

        return xarray_module.speed(u, v)
    return array.speed(u, v)


def direction(u: Any, v: Any, convention: str = "meteo", to_positive: bool = True) -> Any:
    r"""Compute the direction/angle of a vector quantity.

    Parameters
    ----------
    u: array-like or xarray.DataArray
        u wind/x vector component
    v: array-like or xarray.DataArray
        v wind/y vector component (same units as ``u``)
    convention: str, optional
        Specify how the direction/angle is interpreted. The possible values are as follows:

        * "meteo": the direction is the meteorological wind direction
        * "polar": the direction is measured anti-clockwise from the x axis (East/right) to the vector

    to_positive: bool, optional
        If it is True the resulting values are mapped into the [0, 360] range when
        ``convention`` is "polar". Otherwise they lie in the [-180, 180] range.

    Returns
    -------
    array-like or xarray.DataArray
        Direction/angle (degrees)

    The meteorological wind direction is the direction from which the wind is
    blowing. Wind direction increases clockwise such that a northerly wind is 0°, an easterly
    wind is 90°, a southerly wind is 180°, and a westerly wind is 270°.

    In the "polar" convention the direction is measured anti-clockwise from the x axis
    (East/right) to the vector. When ``to_positive`` is True, angles are mapped to [0, 360];
    otherwise they lie in [-180, 180].

    .. image:: /_static/wind_direction.png
        :width: 400px
    """
    if _is_xarray(u):
        from . import xarray as xarray_module

        return xarray_module.direction(u, v, convention=convention, to_positive=to_positive)
    return array.direction(u, v, convention=convention, to_positive=to_positive)


def xy_to_polar(x: Any, y: Any, convention: str = "meteo") -> tuple[Any, Any]:
    r"""Convert wind/vector data from xy representation to polar representation.

    Parameters
    ----------
    x: array-like or xarray.DataArray
        u wind/x vector component
    y: array-like or xarray.DataArray
        v wind/y vector component (same units as ``x``)
    convention: str
        Specify how the direction/angle component of the target polar coordinate
        system is interpreted. The possible values are as follows:

        * "meteo": the direction is the meteorological wind direction
        * "polar": the direction is measured anti-clockwise from the x axis (East/right) to the vector

    Returns
    -------
    array-like or xarray.DataArray
        Magnitude (same units as ``x``)
    array-like or xarray.DataArray
        Direction (degrees)

    In the target xy representation the x axis points East while the y axis points North.
    """
    if _is_xarray(x):
        from . import xarray as xarray_module

        return xarray_module.xy_to_polar(x, y, convention=convention)
    return array.xy_to_polar(x, y, convention=convention)


def polar_to_xy(magnitude: Any, direction: Any, convention: str = "meteo") -> tuple[Any, Any]:
    r"""Convert wind/vector data from polar representation to xy representation.

    Parameters
    ----------
    magnitude: array-like or xarray.DataArray
        Speed/magnitude of the vector
    direction: array-like or xarray.DataArray
        Direction of the vector (degrees)
    convention: str
        Specify how ``direction`` is interpreted. The possible values are as follows:

        * "meteo": ``direction`` is the meteorological wind direction
        * "polar": ``direction`` is the angle measured anti-clockwise from the x axis
          (East/right) to the vector

    Returns
    -------
    array-like or xarray.DataArray
        X vector component (same units as ``magnitude``)
    array-like or xarray.DataArray
        Y vector component (same units as ``magnitude``)

    In the target xy representation the x axis points East while the y axis points North.
    """
    if _is_xarray(magnitude):
        from . import xarray as xarray_module

        return xarray_module.polar_to_xy(magnitude, direction, convention=convention)
    return array.polar_to_xy(magnitude, direction, convention=convention)


def w_from_omega(omega: Any, t: Any, p: Any) -> Any:
    r"""Compute the hydrostatic vertical velocity from pressure velocity.

    Parameters
    ----------
    omega : array-like or xarray.DataArray
        Hydrostatic pressure velocity (Pa/s)
    t : array-like or xarray.DataArray
        Temperature (K)
    p : array-like or xarray.DataArray
        Pressure (Pa)

    Returns
    -------
    array-like or xarray.DataArray
        Hydrostatic vertical velocity (m/s)

    The computation is based on the following hydrostatic formula:

    .. math::

        w = - \frac{\omega\; t R_{d}}{p g}

    where

        * :math:`R_{d}` is the specific gas constant for dry air (see :data:`earthkit.meteo.constants.Rd`).
        * :math:`g` is the gravitational acceleration (see :data:`earthkit.meteo.constants.g`).
    """
    if _is_xarray(omega):
        from . import xarray as xarray_module

        return xarray_module.w_from_omega(omega, t, p)
    return array.w_from_omega(omega, t, p)


def coriolis(lat: Any) -> Any:
    r"""Compute the Coriolis parameter.

    Parameters
    ----------
    lat : array-like or xarray.DataArray
        Latitude (degrees)

    Returns
    -------
    array-like or xarray.DataArray
        The Coriolis parameter (:math:`s^{-1}`)

    The Coriolis parameter is defined by the following formula:

    .. math::

        f = 2 \Omega sin(\phi)

    where :math:`\Omega` is the rotation rate of Earth
    (see :data:`earthkit.meteo.constants.omega`) and :math:`\phi` is the latitude.
    """
    if _is_xarray(lat):
        from . import xarray as xarray_module

        return xarray_module.coriolis(lat)
    return array.coriolis(lat)


def windrose(
    speed: Any,
    direction: Any,
    sectors: int = 16,
    speed_bins: Iterable[float] | None = None,
    percent: bool = True,
) -> tuple[Any, Any]:
    r"""Generate windrose data.

    Parameters
    ----------
    speed : array-like or xarray.DataArray
        Speed
    direction : array-like or xarray.DataArray
        Meteorological wind direction (degrees). See :func:`direction` for details.
        Values must be between 0 and 360.
    sectors: number
        Number of sectors the 360 degrees direction range is split into.
    speed_bin: array-like
        Speed bins
    percent: bool
        If False, returns the number of valid samples in each bin. If True, returns
        the percentage of the number of samples in each bin with respect to the total
        number of valid samples.

    Returns
    -------
    array-like or xarray.DataArray
        The bi-dimensional histogram of ``speed`` and ``direction``.  Values in
        ``speed`` are histogrammed along the first dimension and values in ``direction``
        are histogrammed along the second dimension.
    array-like or xarray.DataArray
        The direction bins (i.e. the sectors) (degrees)

    The sectors do not start at 0 degrees (North) but are shifted by half a sector size.
    E.g. if ``sectors`` is 4 the sectors are defined as:

    .. image:: /_static/wind_sector.png
        :width: 350px
    """
    if _is_xarray(speed):
        from . import xarray as xarray_module

        return xarray_module.windrose(
            speed,
            direction,
            sectors=sectors,
            speed_bins=speed_bins,
            percent=percent,
        )
    return array.windrose(
        speed,
        direction,
        sectors=sectors,
        speed_bins=speed_bins,
        percent=percent,
    )
