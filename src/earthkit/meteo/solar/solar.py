# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, TypeAlias, overload

from earthkit.meteo.utils.decorators import _is_xarray

from . import array

if TYPE_CHECKING:
    import datetime

    import xarray  # type: ignore[import]

ArrayLike: TypeAlias = Any


def _dispatch_if_xarray(func_name: str, *args: Any, **kwargs: Any) -> Any | None:
    # Solar functions often take `date` as the first argument (scalar datetime),
    # while lat/lon are xarray objects. The default `dispatch()` only checks args[0],
    # so we do a custom dispatch: if *any* positional arg is xarray, use xarray impl.
    if any(_is_xarray(a) for a in args):
        module = import_module(__name__.rsplit(".", 1)[0] + ".xarray")
        return getattr(module, func_name)(*args, **kwargs)
    return None


@overload
def julian_day(date: "xarray.DataArray") -> "xarray.DataArray": ...
@overload
def julian_day(date: "datetime.datetime") -> float: ...
def julian_day(date):
    r = _dispatch_if_xarray("julian_day", date)
    return r if r is not None else array.julian_day(date)


@overload
def solar_declination_angle(date: "xarray.DataArray") -> tuple["xarray.DataArray", "xarray.DataArray"]: ...
@overload
def solar_declination_angle(date: "datetime.datetime") -> tuple[float, float]: ...
def solar_declination_angle(date):
    r = _dispatch_if_xarray("solar_declination_angle", date)
    return r if r is not None else array.solar_declination_angle(date)


@overload
def cos_solar_zenith_angle(
    date: "datetime.datetime", latitudes: "xarray.DataArray", longitudes: "xarray.DataArray"
) -> "xarray.DataArray": ...
@overload
def cos_solar_zenith_angle(date: "datetime.datetime", latitudes: ArrayLike, longitudes: ArrayLike): ...
def cos_solar_zenith_angle(date, latitudes, longitudes):
    r = _dispatch_if_xarray("cos_solar_zenith_angle", date, latitudes, longitudes)
    return r if r is not None else array.cos_solar_zenith_angle(date, latitudes, longitudes)


@overload
def cos_solar_zenith_angle_integrated(
    begin_date: "datetime.datetime",
    end_date: "datetime.datetime",
    latitudes: "xarray.DataArray",
    longitudes: "xarray.DataArray",
    *,
    intervals_per_hour: int = 1,
    integration_order: int = 3,
) -> "xarray.DataArray": ...
@overload
def cos_solar_zenith_angle_integrated(
    begin_date: "datetime.datetime",
    end_date: "datetime.datetime",
    latitudes: ArrayLike,
    longitudes: ArrayLike,
    *,
    intervals_per_hour: int = 1,
    integration_order: int = 3,
): ...
def cos_solar_zenith_angle_integrated(
    begin_date,
    end_date,
    latitudes,
    longitudes,
    *,
    intervals_per_hour: int = 1,
    integration_order: int = 3,
):
    r = _dispatch_if_xarray(
        "cos_solar_zenith_angle_integrated",
        begin_date,
        end_date,
        latitudes,
        longitudes,
        intervals_per_hour=intervals_per_hour,
        integration_order=integration_order,
    )
    return (
        r
        if r is not None
        else array.cos_solar_zenith_angle_integrated(
            begin_date,
            end_date,
            latitudes,
            longitudes,
            intervals_per_hour=intervals_per_hour,
            integration_order=integration_order,
        )
    )


@overload
def incoming_solar_radiation(date: "xarray.DataArray") -> "xarray.DataArray": ...
@overload
def incoming_solar_radiation(date: "datetime.datetime") -> float: ...
def incoming_solar_radiation(date):
    r = _dispatch_if_xarray("incoming_solar_radiation", date)
    return r if r is not None else array.incoming_solar_radiation(date)


@overload
def toa_incident_solar_radiation(
    begin_date: "datetime.datetime",
    end_date: "datetime.datetime",
    latitudes: "xarray.DataArray",
    longitudes: "xarray.DataArray",
    *,
    intervals_per_hour: int = 1,
    integration_order: int = 3,
) -> "xarray.DataArray": ...
@overload
def toa_incident_solar_radiation(
    begin_date: "datetime.datetime",
    end_date: "datetime.datetime",
    latitudes: ArrayLike,
    longitudes: ArrayLike,
    *,
    intervals_per_hour: int = 1,
    integration_order: int = 3,
): ...
def toa_incident_solar_radiation(
    begin_date,
    end_date,
    latitudes,
    longitudes,
    *,
    intervals_per_hour: int = 1,
    integration_order: int = 3,
):
    r = _dispatch_if_xarray(
        "toa_incident_solar_radiation",
        begin_date,
        end_date,
        latitudes,
        longitudes,
        intervals_per_hour=intervals_per_hour,
        integration_order=integration_order,
    )
    return (
        r
        if r is not None
        else array.toa_incident_solar_radiation(
            begin_date,
            end_date,
            latitudes,
            longitudes,
            intervals_per_hour=intervals_per_hour,
            integration_order=integration_order,
        )
    )