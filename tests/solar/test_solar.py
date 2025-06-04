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

from earthkit.meteo import solar
from earthkit.meteo.utils.testing import ARRAY_BACKENDS


@pytest.mark.parametrize(
    "date,expected_value",
    [
        (datetime.datetime(2024, 4, 22), 112.0),
        (datetime.datetime(2024, 4, 22, 12, 0, 0), 112.5),
        (datetime.datetime(2024, 4, 22, 12, tzinfo=datetime.timezone(datetime.timedelta(hours=1))), 112.5),
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


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "date,lat,lon,v_ref",
    [(datetime.datetime(2024, 4, 22, 12, 0, 0), 40.0, 18.0, 0.8478445449796352)],
)
def test_cos_solar_zenith_angle_1(date, lat, lon, v_ref, array_backend):
    lat, lon, v_ref = array_backend.asarray(lat, lon, v_ref)
    v = solar.cos_solar_zenith_angle(date, lat, lon)
    v_ref = array_backend.asarray(v_ref, dtype=v.dtype)
    assert array_backend.allclose(v, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "begin_date,end_date,lat,lon,integration_order,v_ref",
    [
        (datetime.datetime(2024, 4, 22), datetime.datetime(2024, 4, 23), 40.0, 18.0, 1, 0.3110738757),
        (datetime.datetime(2024, 4, 22), datetime.datetime(2024, 4, 23), 40.0, 18.0, 2, 0.3110738757),
        (datetime.datetime(2024, 4, 22), datetime.datetime(2024, 4, 23), 40.0, 18.0, 3, 0.3110738757),
        (datetime.datetime(2024, 4, 22), datetime.datetime(2024, 4, 23), 40.0, 18.0, 4, 0.3110738757),
    ],
)
def test_cos_solar_zenith_angle_integrated(
    begin_date, end_date, lat, lon, integration_order, v_ref, array_backend
):
    lat, lon, v_ref = array_backend.asarray(lat, lon, v_ref)
    v = solar.cos_solar_zenith_angle_integrated(
        begin_date, end_date, lat, lon, integration_order=integration_order
    )
    v_ref = array_backend.asarray(v_ref, dtype=v.dtype)
    assert array_backend.allclose(v, v_ref)


def test_incoming_solar_radiation():
    date = datetime.datetime(2024, 4, 22, 12, 0, 0)
    v = solar.incoming_solar_radiation(date)
    assert np.isclose(v, 4833557.3088814365)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "begin_date,end_date,lat,lon,v_ref",
    [(datetime.datetime(2024, 4, 22), datetime.datetime(2024, 4, 23), 40.0, 18.0, 1503617.8237746414)],
)
def test_toa_incident_solar_radiation(begin_date, end_date, lat, lon, v_ref, array_backend):
    lat, lon, v_ref = array_backend.asarray(lat, lon, v_ref)
    v = solar.toa_incident_solar_radiation(begin_date, end_date, lat, lon)
    assert array_backend.allclose(v, v_ref)
