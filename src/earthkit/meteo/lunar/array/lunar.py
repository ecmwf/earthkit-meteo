# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
)

from earthkit.utils.array import array_namespace

ArrayLike: TypeAlias = Any
ArrayNamespace: TypeAlias = Any

_ASTROPY_AVAILABLE = True
try:
    import astropy.units as astropy_units
    from astropy.coordinates import ITRS, EarthLocation, get_body
    from astropy.time import Time
except ImportError:
    _ASTROPY_AVAILABLE = False


def _require_astropy(function_name: str):
    if not _ASTROPY_AVAILABLE:
        raise ImportError(f"`astropy` is required for the function `{function_name}`.")


if TYPE_CHECKING:
    import astropy  # type: ignore[import]


def _get_body_xyz(body_name: str, time: "astropy.time.Time", xp: ArrayNamespace) -> Any:
    """Get the geocentric cartesian coordinates of a celestial body in km.

    Parameters
    ----------
    body_name : str
        Name of the celestial body (e.g., 'moon', 'earth').
    time : astropy.time.Time
        The time at which to compute the position.
    xp : ArrayNamespace
        The array namespace (e.g., numpy, cupy).

    Returns
    -------
    xyz : array-like (shape (3,))
        Geocentric cartesian coordinates of the body in km.
    """
    body = get_body(body_name, time)
    body_itrs = body.transform_to(ITRS(obstime=time))

    xyz = xp.asarray([
        body_itrs.cartesian.x.to(astropy_units.km).value,
        body_itrs.cartesian.y.to(astropy_units.km).value,
        body_itrs.cartesian.z.to(astropy_units.km).value,
    ])

    return xyz


def _get_observer_xyz(
    time: "astropy.time.Time", latitudes: ArrayLike, longitudes: ArrayLike, xp: ArrayNamespace
) -> Any:
    """Get the ITRS cartesian coordinates of surface observers on Earth at a given time.

    Parameters
    ----------
    time : astropy.time.Time
        The observation time (used to set the ITRS obstime).
    latitudes : array-like
        Latitudes of the observer(s) in degrees.
    longitudes : array-like
        Longitudes of the observer(s) in degrees.
    xp : ArrayNamespace
        The array namespace (e.g., numpy, cupy).

    Returns
    -------
    xyz : array-like (shape (3, N))
        ITRS cartesian coordinates of the observer(s) in km.
    """
    loc = EarthLocation.from_geodetic(
        lon=longitudes * astropy_units.deg, lat=latitudes * astropy_units.deg, height=0 * astropy_units.m
    )
    obs_itrs = loc.get_itrs(obstime=time)
    obs_xyz = xp.asarray([
        obs_itrs.cartesian.x.to(astropy_units.km).value,
        obs_itrs.cartesian.y.to(astropy_units.km).value,
        obs_itrs.cartesian.z.to(astropy_units.km).value,
    ])  # shape (3, N)

    return obs_xyz


def _get_distance_between_bodies(
    observer_xyz: ArrayLike, target_xyz: ArrayLike, xp: ArrayNamespace, device: Any
) -> Any:
    """Compute the distance from observer(s) to a target body.

    Parameters
    ----------
    observer_xyz : array-like (shape (3, N) or (3,))
        Cartesian coordinates of the observer(s) in km.
    target_xyz : array-like (shape (3,) or (3, N))
        Cartesian coordinates of the target body in km.
    xp : ArrayNamespace
        The array namespace (e.g., numpy, cupy).
    device : device
        The device on which to return the array.

    Returns
    -------
    distances : array-like (shape (N,))
        Distances from observer(s) to the target body in km.
    """
    if observer_xyz.ndim == 1:
        observer_xyz = observer_xyz[:, xp.newaxis]  # shape (3, 1)
    if target_xyz.ndim == 1:
        target_xyz = target_xyz[:, xp.newaxis]  # shape (3, 1)
    diff = observer_xyz - target_xyz  # shape (3, N)
    distances = xp.asarray(xp.linalg.norm(diff, axis=0), device=device)  # shape (N,)

    return distances


def singular_distance_to_moon(date: datetime.datetime, latitudes: ArrayLike, longitudes: ArrayLike) -> Any:
    """Distance to the Moon in km from the Earth centre,
    with no reference to the latitude and longitude of the observer.

    Parameters
    ----------
    date : datetime.datetime
        The date and time for which to compute the distance.
    latitudes : array-like
        Latitudes, used only for array namespace and device inference.
    longitudes : array-like
        Longitudes, used only for array namespace and device inference.

    Returns
    -------
    distance : float
        Distance to the Moon in km from the Earth centre at the given date and time.
    """
    _require_astropy("distance_to_moon")

    xp = array_namespace(latitudes, longitudes)
    device = xp.device(latitudes)

    time = Time(date)  # Convert to astropy Time object

    moon_xyz = _get_body_xyz("moon", time, xp)
    earth_xyz = _get_body_xyz("earth", time, xp)
    distance = _get_distance_between_bodies(earth_xyz, moon_xyz, xp, device)

    return distance


def distance_to_moon(date: datetime.datetime, latitudes: ArrayLike, longitudes: ArrayLike) -> Any:
    """Distance to the Moon in km.

    Parameters
    ----------
    date : datetime.datetime
        The date and time for which to compute the distance.
    latitudes : array-like
        Latitudes of the observer(s) in degrees.
    longitudes : array-like
        Longitudes of the observer(s) in degrees.

    Returns
    -------
    distances : array-like
        Distances to the Moon in km.
    """
    _require_astropy("distance_to_moon")

    xp = array_namespace(latitudes, longitudes)
    latitudes = xp.asarray(latitudes)
    longitudes = xp.asarray(longitudes)
    device = xp.device(latitudes)

    time = Time(date)  # Convert to astropy Time object

    moon_xyz = _get_body_xyz("moon", time, xp)
    observer_xyz = _get_observer_xyz(time, latitudes, longitudes, xp)

    distances = _get_distance_between_bodies(observer_xyz, moon_xyz, xp, device)
    return distances


def delta_distance_to_moon(date: datetime.datetime, latitudes: ArrayLike, longitudes: ArrayLike) -> Any:
    """Delta distance to the Moon in km, relative to the minimum instantaneous distance.

    Parameters
    ----------
    date : datetime.datetime
        The date and time for which to compute the delta distance.
    latitudes : array-like
        Latitudes of the observer(s) in degrees.
    longitudes : array-like
        Longitudes of the observer(s) in degrees.

    Returns
    -------
    delta_distances : array-like
        The difference between the distances and the minimum distance to the Moon of the specific observer(s).
    """
    distances = distance_to_moon(date, latitudes, longitudes)
    xp = array_namespace(distances)
    min_distance = xp.min(distances)
    delta_distances = distances - min_distance

    return delta_distances
