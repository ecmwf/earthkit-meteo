# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

import xarray as xr

from earthkit.meteo.utils.decorators import xarray_ufunc

from .. import array


def julian_day(date: xr.DataArray) -> xr.DataArray:
    r"""Compute the Julian day (day of year as a fractional number).

    Parameters
    ----------
    date: xarray.DataArray
        Date/time. Computation is performed element-wise.

    Returns
    -------
    xarray.DataArray
        Day of year as a fractional number. January 1st at 00:00 is 0.0.
    """
    return xarray_ufunc(array.julian_day, date, xarray_ufunc_kwargs={"vectorize": True})


def solar_declination_angle(date: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    r"""Compute the solar declination angle and time correction.

    Parameters
    ----------
    date: xarray.DataArray
        Date/time. Computation is performed element-wise.

    Returns
    -------
    xarray.DataArray
        Solar declination angle (degrees).
    xarray.DataArray
        Time correction (degrees).

    Notes
    -----
    This function returns two outputs. We explicitly provide ``output_dtypes`` to
    ensure xarray correctly allocates the outputs when vectorizing.
    """
    return xarray_ufunc(
        array.solar_declination_angle,
        date,
        xarray_ufunc_kwargs={
            "vectorize": True,
            "output_dtypes": [float, float],
        },
    )


def cos_solar_zenith_angle(date, latitudes: xr.DataArray, longitudes: xr.DataArray) -> xr.DataArray:
    r"""Compute the cosine of the solar zenith angle.

    Parameters
    ----------
    date: datetime.datetime
        Date/time (typically a scalar applying to all latitude/longitude points).
    latitudes: xarray.DataArray
        Latitude (degrees).
    longitudes: xarray.DataArray
        Longitude (degrees).

    Returns
    -------
    xarray.DataArray
        Cosine of the solar zenith angle (clipped to be non-negative).

    Notes
    -----
    The result is clipped to the [0, 1] range by setting negative values to 0.
    """

    def _impl(lat, lon):
        return array.cos_solar_zenith_angle(date, lat, lon)

    return xarray_ufunc(_impl, latitudes, longitudes)


def cos_solar_zenith_angle_integrated(
    begin_date,
    end_date,
    latitudes: xr.DataArray,
    longitudes: xr.DataArray,
    *,
    intervals_per_hour: int = 1,
    integration_order: int = 3,
) -> xr.DataArray:
    r"""Compute the time-integrated cosine of the solar zenith angle.

    Parameters
    ----------
    begin_date: datetime.datetime
        Start of the integration interval.
    end_date: datetime.datetime
        End of the integration interval.
    latitudes: xarray.DataArray
        Latitude (degrees).
    longitudes: xarray.DataArray
        Longitude (degrees).
    intervals_per_hour: int, optional
        Number of sub-intervals per hour used in the numerical integration.
    integration_order: int, optional
        Order of the integration scheme.

    Returns
    -------
    xarray.DataArray
        Time-integrated cosine of the solar zenith angle.
    """

    def _impl(lat, lon):
        return array.cos_solar_zenith_angle_integrated(
            begin_date,
            end_date,
            lat,
            lon,
            intervals_per_hour=intervals_per_hour,
            integration_order=integration_order,
        )

    return xarray_ufunc(_impl, latitudes, longitudes)


def incoming_solar_radiation(date: xr.DataArray) -> xr.DataArray:
    r"""Compute the incoming solar radiation at the top of the atmosphere (TOA).

    Parameters
    ----------
    date: xarray.DataArray
        Date/time. Computation is performed element-wise.

    Returns
    -------
    xarray.DataArray
        Incoming solar radiation at TOA.
    """
    return xarray_ufunc(array.incoming_solar_radiation, date, xarray_ufunc_kwargs={"vectorize": True})


def toa_incident_solar_radiation(
    begin_date,
    end_date,
    latitudes: xr.DataArray,
    longitudes: xr.DataArray,
    *,
    intervals_per_hour: int = 1,
    integration_order: int = 3,
) -> xr.DataArray:
    r"""Compute the time-integrated incident solar radiation at the top of the atmosphere (TOA).

    Parameters
    ----------
    begin_date: datetime.datetime
        Start of the integration interval.
    end_date: datetime.datetime
        End of the integration interval.
    latitudes: xarray.DataArray
        Latitude (degrees).
    longitudes: xarray.DataArray
        Longitude (degrees).
    intervals_per_hour: int, optional
        Number of sub-intervals per hour used in the numerical integration.
    integration_order: int, optional
        Order of the integration scheme.

    Returns
    -------
    xarray.DataArray
        Time-integrated incident solar radiation at TOA.
    """

    def _impl(lat, lon):
        return array.toa_incident_solar_radiation(
            begin_date,
            end_date,
            lat,
            lon,
            intervals_per_hour=intervals_per_hour,
            integration_order=integration_order,
        )

    return xarray_ufunc(_impl, latitudes, longitudes)
