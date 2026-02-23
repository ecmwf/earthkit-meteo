# (C) Copyright 2021 ECMWF.
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
    # Elementwise over `date` (typically a time coordinate/array).
    # NOTE: if `date` is numpy.datetime64, the underlying array implementation
    # must support that (or convert).
    return xarray_ufunc(array.julian_day, date, xarray_ufunc_kwargs={"vectorize": True})


def solar_declination_angle(date: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    return xarray_ufunc(array.solar_declination_angle, date, xarray_ufunc_kwargs={"vectorize": True})


def cos_solar_zenith_angle(date, latitudes: xr.DataArray, longitudes: xr.DataArray) -> xr.DataArray:
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