# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import datetime

import numpy as np
import pytest
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo.solar.array import solar


@pytest.mark.parametrize(
    "date,expected_value",
    [
        (datetime.datetime(2024, 4, 22), 112.0),
        (datetime.datetime(2024, 4, 22, 12, 0, 0), 112.5),
        (
            datetime.datetime(2024, 4, 22, 12, tzinfo=datetime.timezone(datetime.timedelta(hours=1))),
            112.5,
        ),
    ],
)
def test_julian_day(date, expected_value):
    v = solar.julian_day(date)

    assert np.isclose(v, expected_value)


@pytest.mark.parametrize(
    "date,v_ref",
    [
        (datetime.datetime(2024, 4, 22), (12.235799080498582, 0.40707190497656276)),
        (
            datetime.datetime(2024, 4, 22, 12, 0, 0),
            (12.403019177270453, 0.43253901867797273),
        ),
    ],
)
def test_solar_declination_angle(date, v_ref):
    declination, time_correction = solar.solar_declination_angle(date)
    assert np.isclose(declination, v_ref[0])
    assert np.isclose(time_correction, v_ref[1])


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "date,lat,lon,v_ref",
    [(datetime.datetime(2024, 4, 22, 12, 0, 0), 40.0, 18.0, 0.8478445449796352)],
)
def test_cos_solar_zenith_angle_1(xp, device, date, lat, lon, v_ref):
    lat = xp.asarray(lat, device=device)
    lon = xp.asarray(lon, device=device)
    v_ref = xp.asarray(v_ref, device=device)
    v = solar.cos_solar_zenith_angle(date, lat, lon)
    v_ref = xp.asarray(v_ref, dtype=v.dtype)
    assert xp.allclose(v, v_ref)


def test_cos_solar_zenith_angle_uses_minutes():
    # The hour angle advances 15 degrees an hour, so moving the time forward by 30
    # minutes moves the sun as moving the place 7.5 degrees east does. The declination
    # changes slightly in 30 minutes, hence the tolerance.
    date = datetime.datetime(2024, 4, 22, 9, 0, 0)
    later = solar.cos_solar_zenith_angle(date + datetime.timedelta(minutes=30), 40.0, 18.0)
    east = solar.cos_solar_zenith_angle(date, 40.0, 18.0 + 7.5)
    at_the_hour = solar.cos_solar_zenith_angle(date, 40.0, 18.0)
    assert np.isclose(later, east, atol=1e-4)
    assert not np.isclose(later, at_the_hour, atol=1e-2)


def test_cos_solar_zenith_angle_integrated_over_a_sunrise_hour():
    # In London on 21 June 2025 the sun rises at about 03:43 UTC. Over the hour from
    # 03:00 to 04:00 the sun is up for about 17 minutes, so the average is small but
    # positive. It must match a fine average of the instantaneous value.
    begin = datetime.datetime(2025, 6, 21, 3, 0, 0)
    end = datetime.datetime(2025, 6, 21, 4, 0, 0)
    v = solar.cos_solar_zenith_angle_integrated(begin, end, 51.5, -0.1, intervals_per_hour=4)
    fine = np.mean([
        solar.cos_solar_zenith_angle(begin + datetime.timedelta(seconds=s + 0.5), 51.5, -0.1) for s in range(3600)
    ])
    assert v > 0.0
    assert np.isclose(v, fine, atol=5e-4)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "begin_date,end_date,lat,lon,integration_order,v_ref",
    [
        (
            datetime.datetime(2024, 4, 22),
            datetime.datetime(2024, 4, 23),
            40.0,
            18.0,
            1,
            0.3108234014,
        ),
        (
            datetime.datetime(2024, 4, 22),
            datetime.datetime(2024, 4, 23),
            40.0,
            18.0,
            2,
            0.3112472832,
        ),
        (
            datetime.datetime(2024, 4, 22),
            datetime.datetime(2024, 4, 23),
            40.0,
            18.0,
            3,
            0.3109985566,
        ),
        (
            datetime.datetime(2024, 4, 22),
            datetime.datetime(2024, 4, 23),
            40.0,
            18.0,
            4,
            0.3111356893,
        ),
    ],
)
def test_cos_solar_zenith_angle_integrated(xp, device, begin_date, end_date, lat, lon, integration_order, v_ref):
    lat = xp.asarray(lat, device=device)
    lon = xp.asarray(lon, device=device)
    v_ref = xp.asarray(v_ref, device=device)
    v = solar.cos_solar_zenith_angle_integrated(begin_date, end_date, lat, lon, integration_order=integration_order)
    v_ref = xp.asarray(v_ref, dtype=v.dtype)
    assert xp.allclose(v, v_ref)


def test_incoming_solar_radiation():
    date = datetime.datetime(2024, 4, 22, 12, 0, 0)
    v = solar.incoming_solar_radiation(date)
    assert np.isclose(v, 4833557.3088814365)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "begin_date,end_date,lat,lon,v_ref",
    [
        (
            datetime.datetime(2024, 4, 22),
            datetime.datetime(2024, 4, 23),
            40.0,
            18.0,
            1503271.6092681934,
        )
    ],
)
def test_toa_incident_solar_radiation(xp, device, begin_date, end_date, lat, lon, v_ref):
    lat = xp.asarray(lat, device=device)
    lon = xp.asarray(lon, device=device)
    v_ref = xp.asarray(v_ref, device=device)
    v = solar.toa_incident_solar_radiation(begin_date, end_date, lat, lon)
    assert xp.allclose(v, v_ref)
