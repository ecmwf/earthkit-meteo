# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import TypeAlias
from typing import overload

from earthkit.meteo.utils.decorators import dispatch

if TYPE_CHECKING:
    import datetime

    import numpy as np
    import xarray  # type: ignore[import]

DateLike: TypeAlias = "datetime.datetime | np.datetime64"

ArrayLike: TypeAlias = Any


@overload
def julian_day(date: "xarray.DataArray") -> "xarray.DataArray": ...
@overload
def julian_day(date: DateLike) -> float: ...
def julian_day(date):
    r"""Compute the Julian day (day of year as a fractional number).

    Parameters
    ----------
    date: datetime.datetime | numpy.datetime64 | xarray.DataArray
        Date/time. When ``date`` is an xarray.DataArray it is processed element-wise.

    Returns
    -------
    float | xarray.DataArray
        Day of year as a fractional number. January 1st at 00:00 is 0.0.

    Implementations
    ------------------------
    :func:`julian_day` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.solar.xarray.julian_day` for xarray.DataArray
    - :py:meth:`earthkit.meteo.solar.array.julian_day` for scalar/array inputs

    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(julian_day, array=True)
    return dispatched(date)


@overload
def solar_declination_angle(date: "xarray.DataArray") -> tuple["xarray.DataArray", "xarray.DataArray"]: ...
@overload
def solar_declination_angle(date: DateLike) -> tuple[float, float]: ...
def solar_declination_angle(date):
    r"""Compute the solar declination angle and time correction.

    Parameters
    ----------
    date: datetime.datetime | numpy.datetime64 | xarray.DataArray
        Date/time.

    Returns
    -------
    (float, float) | (xarray.DataArray, xarray.DataArray)
        Tuple of

        * solar declination angle (degrees)
        * time correction (degrees)

    Implementations
    ------------------------
    :func:`solar_declination_angle` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.solar.xarray.solar_declination_angle` for xarray.DataArray
    - :py:meth:`earthkit.meteo.solar.array.solar_declination_angle` for scalar/array inputs

    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(solar_declination_angle, array=True)
    return dispatched(date)


@overload
def cos_solar_zenith_angle(
    date: DateLike, latitudes: "xarray.DataArray", longitudes: "xarray.DataArray"
) -> "xarray.DataArray": ...
@overload
def cos_solar_zenith_angle(date: DateLike, latitudes: ArrayLike, longitudes: ArrayLike): ...
def cos_solar_zenith_angle(date, latitudes, longitudes):
    r"""Compute the cosine of the solar zenith angle.

    Parameters
    ----------
    date: datetime.datetime | numpy.datetime64
        Date/time (typically a scalar applying to all latitude/longitude points).
    latitudes: array-like | xarray.DataArray
        Latitude (degrees).
    longitudes: array-like | xarray.DataArray
        Longitude (degrees).

    Returns
    -------
    array-like | xarray.DataArray
        Cosine of the solar zenith angle (clipped to be non-negative).

    Notes
    -----
    The result is clipped by setting negative values to 0.

    Implementations
    ------------------------
    :func:`cos_solar_zenith_angle` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.solar.xarray.cos_solar_zenith_angle` when any input is xarray.DataArray
    - :py:meth:`earthkit.meteo.solar.array.cos_solar_zenith_angle` otherwise

    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(cos_solar_zenith_angle, match="latitudes", array=True)
    return dispatched(date, latitudes, longitudes)


@overload
def cos_solar_zenith_angle_integrated(
    begin_date: DateLike,
    end_date: DateLike,
    latitudes: "xarray.DataArray",
    longitudes: "xarray.DataArray",
    *,
    intervals_per_hour: int = 1,
    integration_order: int = 3,
) -> "xarray.DataArray": ...
@overload
def cos_solar_zenith_angle_integrated(
    begin_date: DateLike,
    end_date: DateLike,
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
    r"""Compute the time-integrated cosine of the solar zenith angle.

    Parameters
    ----------
    begin_date: datetime.datetime | numpy.datetime64
        Start of the integration interval.
    end_date: datetime.datetime | numpy.datetime64
        End of the integration interval.
    latitudes: array-like | xarray.DataArray
        Latitude (degrees).
    longitudes: array-like | xarray.DataArray
        Longitude (degrees).
    intervals_per_hour: int, optional
        Number of sub-intervals per hour used in the numerical integration.
    integration_order: int, optional
        Order of the integration scheme.

    Returns
    -------
    array-like | xarray.DataArray
        Time-integrated cosine of the solar zenith angle.

    Implementations
    ------------------------
    :func:`cos_solar_zenith_angle_integrated` calls one of the following implementations depending on the type
    of the input arguments:

    - :py:meth:`earthkit.meteo.solar.xarray.cos_solar_zenith_angle_integrated` when any input is xarray.DataArray
    - :py:meth:`earthkit.meteo.solar.array.cos_solar_zenith_angle_integrated` otherwise

    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(cos_solar_zenith_angle_integrated, match="latitudes", array=True)
    return dispatched(
        begin_date,
        end_date,
        latitudes,
        longitudes,
        intervals_per_hour=intervals_per_hour,
        integration_order=integration_order,
    )


@overload
def incoming_solar_radiation(date: "xarray.DataArray") -> "xarray.DataArray": ...
@overload
def incoming_solar_radiation(date: DateLike) -> float: ...
def incoming_solar_radiation(date):
    r"""Compute the incoming solar radiation at the top of the atmosphere (TOA).

    Parameters
    ----------
    date: datetime.datetime | numpy.datetime64 | xarray.DataArray
        Date/time.

    Returns
    -------
    float | xarray.DataArray
        Incoming solar radiation at TOA.

    Implementations
    ------------------------
    :func:`incoming_solar_radiation` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.solar.xarray.incoming_solar_radiation` for xarray.DataArray
    - :py:meth:`earthkit.meteo.solar.array.incoming_solar_radiation` for scalar/array inputs
      (including ``numpy.datetime64``)

    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(incoming_solar_radiation, array=True)
    return dispatched(date)


@overload
def toa_incident_solar_radiation(
    begin_date: DateLike,
    end_date: DateLike,
    latitudes: "xarray.DataArray",
    longitudes: "xarray.DataArray",
    *,
    intervals_per_hour: int = 1,
    integration_order: int = 3,
) -> "xarray.DataArray": ...
@overload
def toa_incident_solar_radiation(
    begin_date: DateLike,
    end_date: DateLike,
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
    r"""Compute the time-integrated incident solar radiation at the top of the atmosphere (TOA).

    Parameters
    ----------
    begin_date: datetime.datetime | numpy.datetime64
        Start of the integration interval.
    end_date: datetime.datetime | numpy.datetime64
        End of the integration interval.
    latitudes: array-like | xarray.DataArray
        Latitude (degrees).
    longitudes: array-like | xarray.DataArray
        Longitude (degrees).
    intervals_per_hour: int, optional
        Number of sub-intervals per hour used in the numerical integration.
    integration_order: int, optional
        Order of the integration scheme.

    Returns
    -------
    array-like | xarray.DataArray
        Time-integrated incident solar radiation at TOA.

    Implementations
    ------------------------
    :func:`toa_incident_solar_radiation` calls one of the following implementations depending on the type
    of the input arguments:

    - :py:meth:`earthkit.meteo.solar.xarray.toa_incident_solar_radiation` when any input is xarray.DataArray
    - :py:meth:`earthkit.meteo.solar.array.toa_incident_solar_radiation` otherwise

    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(toa_incident_solar_radiation, match="latitudes", array=True)
    return dispatched(
        begin_date,
        end_date,
        latitudes,
        longitudes,
        intervals_per_hour=intervals_per_hour,
        integration_order=integration_order,
    )
