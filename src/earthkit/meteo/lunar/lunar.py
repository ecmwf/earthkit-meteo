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
from typing import Any, TypeAlias

from numpy.typing import NDArray

from earthkit.meteo.utils.decorators import dispatch

NDArrayLike: TypeAlias = NDArray | float
ArrayNamespace: TypeAlias = Any


def distance_from_earth_centre_to_moon(date: datetime.datetime) -> float:
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
    from . import array as lunar_array

    return lunar_array.distance_from_earth_centre_to_moon(date)


def distance_to_moon(date: datetime.datetime, latitudes: NDArrayLike, longitudes: NDArrayLike) -> NDArrayLike:
    """Distance to the Moon in km.

    Parameters
    ----------
    date : datetime.datetime
        The date and time for which to compute the distance.
    latitudes : NDArrayLike
        Latitudes of the observer(s) in degrees.
    longitudes : NDArrayLike
        Longitudes of the observer(s) in degrees.

    Returns
    -------
    distances : NDArrayLike
        Distances to the Moon in km.
    """
    dispatched = dispatch(distance_to_moon, match=1, array=True, xarray=False, fieldlist=False)
    return dispatched(date, latitudes, longitudes)


def delta_distance_to_moon(date: datetime.datetime, latitudes: NDArrayLike, longitudes: NDArrayLike) -> NDArrayLike:
    """Delta distance to the Moon in km, relative to the minimum instantaneous distance.

    Parameters
    ----------
    date : datetime.datetime
        The date and time for which to compute the delta distance.
    latitudes : NDArrayLike
        Latitudes of the observer(s) in degrees.
    longitudes : NDArrayLike
        Longitudes of the observer(s) in degrees.

    Returns
    -------
    delta_distances : NDArrayLike
        The difference between the distances and the minimum distance to the Moon of the specific observer(s).
    """
    dispatched = dispatch(delta_distance_to_moon, match=1, array=True, xarray=False, fieldlist=False)
    return dispatched(date, latitudes, longitudes)
