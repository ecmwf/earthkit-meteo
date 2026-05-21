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

from earthkit.data import Field, FieldList  # type: ignore[import]

from .. import array


def cos_solar_zenith_angle(
    date: datetime.datetime,
    data: FieldList | Field,
) -> FieldList | Field:
    r"""Compute the cosine of the solar zenith angle.

    Parameters
    ----------
    date: datetime.datetime
        Date/time.
    data: FieldList|Field
        Field(s) whose geography provides latitude and longitude values.

    Returns
    -------
    FieldList|Field
        Cosine of the solar zenith angle (clipped to be non-negative). The
        result has the same type as the input (FieldList or Field).
    """
    out_metadata = {"parameter.variable": "cossza", "parameter.units": "1"}

    if isinstance(data, Field):
        lat, lon = data.geography.latlons()
        v = array.cos_solar_zenith_angle(date, lat, lon)
        return data.set({"values": v, **out_metadata})

    result = []
    for field in data:
        lat, lon = field.geography.latlons()
        v = array.cos_solar_zenith_angle(date, lat, lon)
        result.append(field.set({"values": v, **out_metadata}))
    return FieldList.from_fields(result)


def cos_solar_zenith_angle_integrated(
    begin_date: datetime.datetime,
    end_date: datetime.datetime,
    data: FieldList | Field,
    *,
    intervals_per_hour: int = 1,
    integration_order: int = 3,
) -> FieldList | Field:
    r"""Compute the time-integrated cosine of the solar zenith angle.

    Parameters
    ----------
    begin_date: datetime.datetime
        Start of the integration interval.
    end_date: datetime.datetime
        End of the integration interval.
    data: FieldList|Field
        Field(s) whose geography provides latitude and longitude values.
    intervals_per_hour: int, optional
        Number of sub-intervals per hour used in the numerical integration.
    integration_order: int, optional
        Order of the Gaussian integration scheme (1, 2, 3, or 4).

    Returns
    -------
    FieldList|Field
        Time-integrated cosine of the solar zenith angle. The result has the
        same type as the input (FieldList or Field).
    """
    out_metadata = {"parameter.variable": "cossza_integrated", "parameter.units": "1"}
    if isinstance(data, Field):
        lat, lon = data.geography.latlons()
        v = array.cos_solar_zenith_angle_integrated(
            begin_date,
            end_date,
            lat,
            lon,
            intervals_per_hour=intervals_per_hour,
            integration_order=integration_order,
        )
        return data.set({"values": v, **out_metadata})

    result = []
    for field in data:
        lat, lon = field.geography.latlons()
        v = array.cos_solar_zenith_angle_integrated(
            begin_date,
            end_date,
            lat,
            lon,
            intervals_per_hour=intervals_per_hour,
            integration_order=integration_order,
        )
        result.append(field.set({"values": v, **out_metadata}))
    return FieldList.from_fields(result)


def toa_incident_solar_radiation(
    begin_date: datetime.datetime,
    end_date: datetime.datetime,
    data: FieldList | Field,
    *,
    intervals_per_hour: int = 1,
    integration_order: int = 3,
) -> FieldList | Field:
    r"""Compute the time-integrated incident solar radiation at the top of the atmosphere (TOA).

    Parameters
    ----------
    begin_date: datetime.datetime
        Start of the integration interval.
    end_date: datetime.datetime
        End of the integration interval.
    data: FieldList|Field
        Field(s) whose geography provides latitude and longitude values.
    intervals_per_hour: int, optional
        Number of sub-intervals per hour used in the numerical integration.
    integration_order: int, optional
        Order of the Gaussian integration scheme (1, 2, 3, or 4).

    Returns
    -------
    FieldList|Field
        Time-integrated incident solar radiation at TOA. The result has the
        same type as the input (FieldList or Field).
    """
    # TODO: clarify the units of the result (W/m^2 * time?) and add to metadata and docstring
    out_metadata = {"parameter.variable": "toa_incident_solar_radiation", "parameter.units": "1"}
    if isinstance(data, Field):
        lat, lon = data.geography.latlons()
        v = array.toa_incident_solar_radiation(
            begin_date,
            end_date,
            lat,
            lon,
            intervals_per_hour=intervals_per_hour,
            integration_order=integration_order,
        )
        return data.set({"values": v, **out_metadata})

    result = []
    for field in data:
        lat, lon = field.geography.latlons()
        v = array.toa_incident_solar_radiation(
            begin_date,
            end_date,
            lat,
            lon,
            intervals_per_hour=intervals_per_hour,
            integration_order=integration_order,
        )
        result.append(field.set({"values": v, **out_metadata}))
    return FieldList.from_fields(result)
