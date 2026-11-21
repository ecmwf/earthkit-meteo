# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TypeAlias

from earthkit.utils.array import array_namespace

from earthkit.meteo import constants

ArrayLike: TypeAlias = Any


def speed(u: ArrayLike, v: ArrayLike) -> ArrayLike:
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
    xp = array_namespace(u, v)
    u = xp.asarray(u)
    v = xp.asarray(v)
    return xp.hypot(u, v)


def _direction_meteo(u: ArrayLike, v: ArrayLike) -> ArrayLike:
    xp = array_namespace(u, v)
    u = xp.asarray(u)
    v = xp.asarray(v)

    minus_pi2 = -xp.pi / 2.0
    d = xp.atan2(v, u)
    d = xp.asarray(d)
    m = d <= minus_pi2
    d[m] = (minus_pi2 - d[m]) * constants.degree
    m = ~m
    d[m] = (1.5 * xp.pi - d[m]) * constants.degree
    return d


def _direction_polar(u: ArrayLike, v: ArrayLike, to_positive: bool) -> ArrayLike:
    xp = array_namespace(u, v)
    u = xp.asarray(u)
    v = xp.asarray(v)
    d = xp.atan2(v, u) * constants.degree
    if to_positive:
        d = xp.asarray(d)
        m = d < 0
        d[m] = 360.0 + d[m]
    return d


def direction(u: ArrayLike, v: ArrayLike, convention: str = "meteo", to_positive: bool = True) -> ArrayLike:
    r"""Compute the direction/angle of a vector quantity.

    Parameters
    ----------
    u: array-like
        u wind/x vector component
    v: array-like
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
    array-like
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

    """
    if convention == "meteo":
        return _direction_meteo(u, v)
    elif convention == "polar":
        return _direction_polar(u, v, to_positive)
    else:
        raise ValueError(f"direction(): invalid convention={convention}!")


def xy_to_polar(x: ArrayLike, y: ArrayLike, convention: str = "meteo") -> tuple[ArrayLike, ArrayLike]:
    r"""Convert wind/vector data from xy representation to polar representation.

    Parameters
    ----------
    x: array-like
        u wind/x vector component
    y: array-like
        v wind/y vector component (same units as ``u``)
    convention: str
        Specify how the direction/angle component of the target polar coordinate
        system is interpreted. The possible values are as follows:

        * "meteo": the direction is the meteorological wind direction (see :func:`direction` for explanation)
        * "polar": the direction is measured anti-clockwise from the x axis (East/right) to the vector


    Returns
    -------
    array-like
        Magnitude (same units as ``u``)
    array-like
        Direction (degrees)

    Notes
    -----
    In the target xy representation the x axis points East while the y axis points North.

    """
    return speed(x, y), direction(x, y, convention=convention)


def _polar_to_xy_meteo(magnitude: ArrayLike, direction: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    xp = array_namespace(magnitude, direction)
    magnitude = xp.asarray(magnitude)
    direction = xp.asarray(direction)

    a = (270.0 - direction) * constants.radian
    return magnitude * xp.cos(a), magnitude * xp.sin(a)


def _polar_to_xy_polar(magnitude: ArrayLike, direction: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    xp = array_namespace(magnitude, direction)
    magnitude = xp.asarray(magnitude)
    direction = xp.asarray(direction)

    a = direction * constants.radian
    return magnitude * xp.cos(a), magnitude * xp.sin(a)


def polar_to_xy(
    magnitude: ArrayLike,
    direction: ArrayLike,
    convention: str = "meteo",
) -> tuple[ArrayLike, ArrayLike]:
    r"""Convert wind/vector data from polar representation to xy representation.

    Parameters
    ----------
    magnitude: array-like
        Speed/magnitude of the vector
    direction: array-like
        Direction of the vector (degrees)
    convention: str
        Specify how ``direction`` is interpreted. The possible values are as follows:

        * "meteo": ``direction`` is the meteorological wind direction
          (see :func:`direction` for explanation)
        * "polar": ``direction`` is the angle measured anti-clockwise from the x axis
          (East/right) to the vector

    Returns
    -------
    array-like
        X vector component (same units as ``magnitude``)
    array-like
        Y vector component (same units as ``magnitude``)


    Notes
    -----
    In the target xy representation the x axis points East while the y axis points North.

    """
    if convention == "meteo":
        return _polar_to_xy_meteo(magnitude, direction)
    elif convention == "polar":
        return _polar_to_xy_polar(magnitude, direction)
    else:
        raise ValueError(f"polar_to_xy(): invalid convention={convention}!")


def w_from_omega(omega: ArrayLike, t: ArrayLike, p: ArrayLike) -> ArrayLike:
    r"""Compute the hydrostatic vertical velocity from pressure velocity, temperature and pressure.

    Parameters
    ----------
    omega : array-like
        Hydrostatic pressure velocity (Pa/s)
    t : array-like
        Temperature (K)
    p : array-like
        Pressure (Pa)

    Returns
    -------
    array-like
        Hydrostatic vertical velocity (m/s)

    Notes
    -----
    The computation is based on the following hydrostatic formula:

    .. math::

        w = - \frac{\omega  t R_{d}}{p g}

    where

        * :math:`R_{d}` is the specific gas constant for dry air (see :data:`earthkit.meteo.constants.Rd`).
        * :math:`g` is the gravitational acceleration (see :data:`earthkit.meteo.constants.g`)

    """
    return (-constants.Rd / constants.g) * (omega * t / p)


def coriolis(lat: ArrayLike) -> ArrayLike:
    r"""Compute the Coriolis parameter.

    Parameters
    ----------
    lat : array-like
        Latitude (degrees)

    Returns
    -------
    array-like
        The Coriolis parameter (:math:`s^{-1}`)

    Notes
    -----
    The Coriolis parameter is defined by the following formula:

    .. math::

        f = 2 \Omega sin(\phi)

    where :math:`\Omega` is the rotation rate of Earth
    (see :data:`earthkit.meteo.constants.omega`) and :math:`\phi` is the latitude.

    """
    xp = array_namespace(lat)
    lat = xp.asarray(lat)
    return 2 * constants.omega * xp.sin(lat * constants.radian)


def windrose(
    speed: ArrayLike,
    direction: ArrayLike,
    sectors: int = 16,
    speed_bins: ArrayLike | None = None,
    percent: bool = True,
    sectors_start: float | None = None,
) -> tuple[ArrayLike, ArrayLike]:
    """Generate windrose data.

    Parameters
    ----------
    speed : array-like
        Speed
    direction : array-like
        Meteorological wind direction (degrees). See :func:`direction` for details.
        Values are expected to be in the [0, 360] range, where 0 and 360 both
        correspond to North. Values outside this range are wrapped into it,
        e.g. -90 is treated as 270.
    sectors : int, optional
        Number of sectors the 360 degrees direction range is split into. Must be at
        least 1. A non-integer value is truncated to an integer. See below for details.
    speed_bins : array-like | None, optional
        Speed bins. Must have at least 2 elements. Samples whose speed is outside the
        [speed_bins[0], speed_bins[-1]] range are ignored.
    percent : bool, optional
        If False, returns the number of samples in each bin. If True, returns the
        percentage of the samples in each bin with respect to the total number of
        samples falling in the histogram.
    sectors_start : float | None, optional
        The starting angle of the first sector (degrees). Must be at least -180 and
        less than 180. If None, the first sector does not start at 0 degrees (North)
        but is centred on it, i.e. ``sectors_start`` is ``-(360 / sectors) / 2``.

        .. versionadded:: 1.4.0


    Returns
    -------
    2d array-like
        The bi-dimensional histogram of ``speed`` and ``direction``.  Values in
        ``speed`` are histogrammed along the first dimension and values in ``direction``
        are histogrammed along the second dimension.

    array-like
        The direction bins (i.e. the sectors) (degrees). It contains ``sectors + 1``
        bin edges, from ``sectors_start`` to ``sectors_start + 360``.


    Notes
    -----
    The ``sectors`` parameter defines the number of direction bins in 360
    degrees. Any integer value of at least 1 is supported. The bins are evenly
    spaced and cover the [``sectors_start``, ``sectors_start + 360``] range. If
    ``sectors_start`` is not provided, then the sectors start at 0 degrees
    (North) shifted by half a sector size, so the first sector is centred on
    North and its lower edge is ``-(360 / sectors) / 2``.

    Each direction is first wrapped into [``sectors_start``, ``sectors_start + 360``)
    and then counted in the bin containing it. Bins are half open,
    [``lower``, ``upper``), except the last one, which also includes its upper edge.
    Since the wrapping happens first, a direction of ``sectors_start + 360`` belongs
    to the first sector, like North.

    E.g. if ``sectors`` is 4 and ``sectors_start`` is not provided, the sector edges
    are -45, 45, 135, 225 and 315 degrees:

    .. image:: /_static/wind_sector.png
        :width: 350px

    If ``sectors_start`` is 0 and ``sectors`` is 4, the sector edges are 0, 90, 180,
    270 and 360 degrees:

    .. image:: /_static/wind_sector_start0.png
        :width: 350px

    While if ``sectors`` is 3, which is also supported, and ``sectors_start`` is not
    provided, they are -60, 60, 180 and 300 degrees.

    Samples where either ``speed`` or ``direction`` is NaN are ignored. If all the
    ``speed`` values are NaN, the histogram is filled with NaN.

    """
    speed_bins = speed_bins if speed_bins is not None else []

    if len(speed_bins) < 2:
        raise ValueError("windrose(): speed_bins must have at least 2 elements!")

    sectors = int(sectors)
    if sectors < 1:
        raise ValueError("windrose(): sectors must be at least 1!")

    xp = array_namespace(speed, direction)

    # TODO: atleast_1d is not part of the array API standard
    speed = xp.atleast_1d(speed)
    direction = xp.atleast_1d(direction)
    device = xp.device(speed)

    dir_step = 360.0 / sectors
    sectors_start = -dir_step / 2 if sectors_start is None else sectors_start
    if not -180 <= sectors_start < 180:
        raise ValueError("windrose(): sectors_start must be at least -180 and less than 180!")

    dir_bins = xp.linspace(sectors_start, sectors_start + 360.0, sectors + 1, dtype=speed.dtype, device=device)
    speed_bins = xp.asarray(speed_bins, dtype=speed.dtype, device=device)

    # If there speed array only contains NaN values, return a histogram filled with NaNs
    if xp.all(xp.isnan(speed)):
        return xp.full((len(speed_bins) - 1, sectors), xp.nan, dtype=speed.dtype, device=device), dir_bins

    # Wrap the directions into [sectors_start, sectors_start + 360). When the bins are floating
    # point, the directions are converted to the dtype of the bins (``speed.dtype``) and
    # ``dir_bins[0]`` is used instead of ``sectors_start``, so that the shift is computed with the
    # same float representation as the histogram edges: adding/subtracting 360 would place a
    # direction at the wrap-around slightly below the first edge (whenever ``sectors_start + 360``
    # is not exactly representable) and a direction with a wider dtype than the edges (e.g. a
    # float64 direction with float32 speed) slightly above the last one. Either sample would be
    # silently dropped by the histogram. With non floating point bins (e.g. an integer speed) a
    # direction cannot be represented by the bins anyway, so the directions are left untouched:
    # converting them would truncate them and turn a NaN direction into a value that is counted.
    floating_bins = xp.isdtype(dir_bins.dtype, "real floating")
    start = dir_bins[0] if floating_bins else sectors_start
    if floating_bins:
        direction = xp.asarray(direction, dtype=dir_bins.dtype, device=device)

    wrapped = xp.remainder(direction - start, 360.0)
    # Rounding may return exactly 360.0 for a direction just below the start. Such a sample is
    # indistinguishable from ``sectors_start`` and belongs to the first sector.
    direction = start + xp.where(wrapped >= 360.0, wrapped - 360.0, wrapped)
    if floating_bins:
        # As a safety net, keep every sample within the edges: rounding could place it just
        # outside them, and the histogram would silently drop it. NaNs are left untouched by clip
        # and are ignored by the histogram.
        direction = xp.clip(direction, dir_bins[0], dir_bins[-1])

    # NOTE: np.histogram2d is only available in numpy. For other namespaces we use a fallback implementation
    # based on histogramdd. (See utils.compute.histogram2d). However, neither histogram2d nor
    # histogramdd are part of the array API standard.
    res = xp.histogram2d(
        speed,
        direction,
        bins=[speed_bins, dir_bins],
    )[0]

    return ((res * 100.0 / res.sum()) if percent else res), dir_bins
