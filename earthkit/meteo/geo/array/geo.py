# (C) Copyright 2021- ECMWF.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

# In applying this licence, ECMWF does not waive the privileges and immunities granted to it by
# virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.


import numpy as np

from earthkit.meteo import constants


def distance(lat1, lon1, lat2, lon2):
    r"""Computes the spherical distance between two (sets of) points on Earth.

    Parameters
    ----------
    lat1: number or ndarray
        Latitudes of first set of points (degrees)
    lon1: number or ndarray
        Longitudes of first set of points (degrees). Must have the same shape as ``lat1``.
    lat2: number or ndarray
        Latitudes of second set of points (degrees)
    lon2: number or ndarray
        Longitudes of second set of points (degrees). Must have the same shape as ``lat2``.

    Returns
    -------
    number or ndarray
        Spherical distance on the surface in Earth (m)


    The computation is based on the following formula:

    .. math::

        d = R_{e}\; arccos(sin\phi_{1} sin\phi_{2} + cos\phi_{1} cos\phi_{2} cos\Delta\lambda)

    where :math:`\phi` and :math:`\lambda` stand for the latitude and longitude,
    and :math:`R_{e}` is the radius of the Earth
    (see :data:`earthkit.meteo.constants.R_earth`).

    As for the input points, the following restrictions apply:

    * at least one of them is a single point
    * or both sets of points have the same shape

    """
    lat1 = np.asarray(lat1)
    lon1 = np.asarray(lon1)
    lat2 = np.asarray(lat2)
    lon2 = np.asarray(lon2)

    if lat1.shape != lon1.shape:
        raise ValueError(
            f"distance: lat1 and lon1 must have the same shape! {lat1.shape} != {lon1.shape}"
        )

    if lat2.shape != lon2.shape:
        raise ValueError(
            f"distance: lat2 and lon2 must have the same shape! {lat2.shape} != {lon2.shape}"
        )

    if lat1.size > 1 and lat2.size > 1 and lat2.shape != lat1.shape:
        raise ValueError(
            f"distance: incompatible shapes! lat1={lat1.shape}, lat2={lat2.shape}"
        )

    lat1_rad = lat1 * constants.radian
    lat2_rad = lat2 * constants.radian

    x = np.sin(lat1_rad) * np.sin(lat2_rad) + np.cos(lat1_rad) * np.cos(
        lat2_rad
    ) * np.cos(constants.radian * (lon2 - lon1))
    np.clip(x, -1, 1, out=x)
    return np.arccos(x) * constants.R_earth


def bearing(lat_ref, lon_ref, lat, lon):
    r"""Computes the bearing with respect to a given reference location.

    Parameters
    ----------
    lat_ref: number or ndarray
        Latitudes of reference points (degrees)
    lon_ref: number or ndarray
        Longitudes of reference points (degrees). Must have the same shape as ``lat_ref``.
    lat: number or ndarray
        Latitudes (degrees)
    lon: number or ndarray
        Longitudes (degrees). Must have the same shape as ``lat``.

    Returns
    -------
    number or ndarray
        Bearing with respect to the reference points (degree).


    The **bearing** is the angle between the Northward meridian going through the reference point
    and the great circle connecting the reference point and the other point. It is measured in
    degrees clockwise from North. If a point is located on the same latitude as the reference
    point the bearing is regarded constant: it is either 90° (East) or 270° (West). If the point
    is co-located with the reference point the bearing is set to np.nan.

    As for the reference and other points, the following restrictions apply:

    * at least one of them is a single point
    * or both the reference and other points have the same shape

    """
    lat_ref = np.asarray(lat_ref)
    lon_ref = np.asarray(lon_ref)
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if lat_ref.shape != lon_ref.shape:
        raise ValueError(
            f"bearing: lat_ref and lon_ref must have the same shape! {lat_ref.shape} != {lon_ref.shape}"
        )

    if lat.shape != lon.shape:
        raise ValueError(
            f"bearing: lat and lon must have the same shape! {lat.shape} != {lon.shape}"
        )

    if lat_ref.size > 1 and lat.size > 1 and lat_ref.shape != lat.shape:
        raise ValueError(
            f"bearing: incompatible shapes! lat_ref={lat_ref.shape}, lat={lat.shape}"
        )

    eps = 1e-9
    if lat_ref.shape:
        br = np.full(lat_ref.shape, np.nan)
    else:
        br = np.full(lat.shape, np.nan)

    # computes longitude difference
    d_lon = lon.copy()
    d_lon[lon > 180.0] -= 360.0
    d_lon = (d_lon - lon_ref) * constants.radian

    lat_ref_rad = lat_ref * constants.radian
    lat_rad = lat * constants.radian

    # the bearing is constant on the same latitude
    mask = np.fabs(lat - lat_ref) < eps
    mask_1 = mask & (np.fabs(d_lon) >= eps)
    br[mask_1 & (d_lon <= 0.0)] = 270.0
    br[mask_1 & (d_lon > 0.0)] = 90.0

    # deals with different latitudes
    mask = ~mask
    if mask.shape == lat_ref.shape:
        lat_ref_rad = lat_ref_rad[mask]
    if mask.shape == lat.shape:
        lat_rad = lat_rad[mask]

    # bearing in degrees, x axis points to East, anti-clockwise
    br_m = np.arctan2(
        np.cos(lat_ref_rad) * np.sin(lat_rad)
        - np.sin(lat_ref_rad) * np.cos(lat_rad) * np.cos(d_lon[mask]),
        np.sin(d_lon[mask]) * np.cos(lat_ref_rad),
    )

    # transforms to the required coordinate system: x axis points to North, clockwise
    br_m = (np.pi / 2.0) - br_m
    br_m[br_m < 0] += 2.0 * np.pi
    br_m *= constants.degree

    br[mask] = br_m
    return br
