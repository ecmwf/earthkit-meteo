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

from earthkit.meteo.utils.decorators import dispatch

ArrayLike: TypeAlias = Any


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
    dispatched = dispatch(singular_distance_to_moon, array=True, xarray=False, fieldlist=False)
    return dispatched(date, latitudes, longitudes)


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
    dispatched = dispatch(distance_to_moon, array=True, xarray=False, fieldlist=False)
    return dispatched(date, latitudes, longitudes)


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
        Delta distances to the Moon in km, relative to the minimum instantaneous distance.
    """
    dispatched = dispatch(delta_distance_to_moon, array=True, xarray=False, fieldlist=False)
    return dispatched(date, latitudes, longitudes)
