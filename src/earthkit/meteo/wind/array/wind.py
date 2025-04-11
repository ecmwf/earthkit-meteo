# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.utils.array import array_namespace

from earthkit.meteo import constants


def speed(u, v):
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


def _direction_meteo(u, v):
    xp = array_namespace(u, v)
    u = xp.asarray(u)
    v = xp.asarray(v)

    minus_pi2 = -xp.pi / 2.0
    d = xp.arctan2(v, u)
    d = xp.asarray(d)
    m = d <= minus_pi2
    d[m] = (minus_pi2 - d[m]) * constants.degree
    m = ~m
    d[m] = (1.5 * xp.pi - d[m]) * constants.degree
    return d


def _direction_polar(u, v, to_positive):
    xp = array_namespace(u, v)
    u = xp.asarray(u)
    v = xp.asarray(v)
    d = xp.arctan2(v, u) * constants.degree
    if to_positive:
        d = xp.asarray(d)
        m = d < 0
        d[m] = 360.0 + d[m]
    return d


def direction(u, v, convention="meteo", to_positive=True):
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

    positive: bool, optional
        If it is True the resulting values are mapped into the [0, 360] range when
        ``convention`` is "polar". Otherwise they lie in the [-180, 180] range.


    Returns
    -------
    array-like
        Direction/angle (degrees)


    The meteorological wind direction is the direction from which the wind is
    blowing. Wind direction increases clockwise such that a northerly wind is 0°, an easterly
    wind is 90°, a southerly wind is 180°, and a westerly wind is 270°. The figure below illustrates
    how it is related to the actual orientation of the wind vector:

    .. image:: /_static/wind_direction.png
        :width: 400px

    """
    if convention == "meteo":
        return _direction_meteo(u, v)
    elif convention == "polar":
        return _direction_polar(u, v, to_positive)
    else:
        raise ValueError(f"direction(): invalid convention={convention}!")


def xy_to_polar(x, y, convention="meteo"):
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


    In the target xy representation the x axis points East while the y axis points North.

    """
    return speed(x, y), direction(x, y, convention=convention)


def _polar_to_xy_meteo(magnitude, direction):
    xp = array_namespace(magnitude, direction)
    magnitude = xp.asarray(magnitude)
    direction = xp.asarray(direction)

    a = (270.0 - direction) * constants.radian
    return magnitude * xp.cos(a), magnitude * xp.sin(a)


def _polar_to_xy_polar(magnitude, direction):
    xp = array_namespace(magnitude, direction)
    magnitude = xp.asarray(magnitude)
    direction = xp.asarray(direction)

    a = direction * constants.radian
    return magnitude * xp.cos(a), magnitude * xp.sin(a)


def polar_to_xy(magnitude, direction, convention="meteo"):
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


    In the target xy representation the x axis points East while the y axis points North.

    """
    if convention == "meteo":
        return _polar_to_xy_meteo(magnitude, direction)
    elif convention == "polar":
        return _polar_to_xy_polar(magnitude, direction)
    else:
        raise ValueError(f"polar_to_xy(): invalid convention={convention}!")


def w_from_omega(omega, t, p):
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


    The computation is based on the following hydrostatic formula:

    .. math::

        w = - \frac{\omega\; t R_{d}}{p g}

    where

        * :math:`R_{d}` is the specific gas constant for dry air (see :data:`earthkit.meteo.constants.Rd`).
        * :math:`g` is the gravitational acceleration (see :data:`earthkit.meteo.constants.g`)

    """
    return (-constants.Rd / constants.g) * (omega * t / p)


def coriolis(lat):
    r"""Compute the Coriolis parameter.

    Parameters
    ----------
    lat : array-like
        Latitude (degrees)

    Returns
    -------
    array-like
        The Coriolis parameter (:math:`s^{-1}`)


    The Coriolis parameter is defined by the following formula:

    .. math::

        f = 2 \Omega sin(\phi)

    where :math:`\Omega` is the rotation rate of Earth
    (see :data:`earthkit.meteo.constants.omega`) and :math:`\phi` is the latitude.

    """
    xp = array_namespace(lat)
    lat = xp.asarray(lat)
    return 2 * constants.omega * xp.sin(lat * constants.radian)


def windrose(speed, direction, sectors=16, speed_bins=None, percent=True):
    """Generate windrose data.

    Parameters
    ----------
    speed : array-like
        Speed
    direction : array-like
        Meteorological wind direction (degrees). See :func:`direction` for details.
        Values must be between 0 and 360.
    sectors: number
        Number of sectors the 360 degrees direction range is split into. See below for details.
    speed_bin: array-like
        Speed bins
    percent: bool
        If False, returns the number of valid samples in each bin. If True, returns
        the percentage of the number of samples in each bin with respect to the total
        number of valid samples.


    Returns
    -------
    2d array-like
       The bi-dimensional histogram of ``speed`` and ``direction``.  Values in
       ``speed`` are histogrammed along the first dimension and values in ``direction``
       are histogrammed along the second dimension.

    array-like
        The direction bins (i.e. the sectors) (degrees)


    The sectors do not start at 0 degrees (North) but are shifted by half a sector size.
    E.g. if ``sectors`` is 4 the sectors are defined as:

    .. image:: /_static/wind_sector.png
        :width: 350px

    """
    speed_bins = speed_bins if speed_bins is not None else []

    if len(speed_bins) < 2:
        raise ValueError("windrose(): speed_bins must have at least 2 elements!")

    sectors = int(sectors)
    if sectors < 1:
        raise ValueError("windrose(): sectors must be greater than 1!")

    xp = array_namespace(speed, direction)

    # TODO: atleast_1d is not part of the array API standard
    speed = xp.atleast_1d(speed)
    direction = xp.atleast_1d(direction)

    dir_step = 360.0 / sectors
    dir_bins = xp.asarray(
        xp.linspace(int(-dir_step / 2), int(360 + dir_step / 2), int(360 / dir_step) + 2), dtype=speed.dtype
    )
    speed_bins = xp.asarray(speed_bins, dtype=speed.dtype)

    # NOTE: np.histogram2d is only available in numpy. For other namespaces we use a fallback implementation
    # based on histogramdd. (See utils.compute.histogram2d). However, neither histogram2d nor
    # histogramdd are part of the array API standard.
    res = xp.histogram2d(
        speed,
        direction,
        bins=[speed_bins, dir_bins],
        density=False,
    )[0]

    # unify the north bins
    res[:, 0] = res[:, 0] + res[:, -1]
    res = res[:, :-1]
    dir_bins = dir_bins[:-1]

    return ((res * 100.0 / res.sum()) if percent else res), dir_bins
