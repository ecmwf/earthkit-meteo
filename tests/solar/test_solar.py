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


@pytest.mark.parametrize(
    "date,expected_value",
    [
        (datetime.datetime(2024, 4, 22), 112.0),
        (datetime.datetime(2024, 4, 22, 12, 0, 0), 112.5),
    ],
)
def test_julian_day(date, expected_value):
    v = solar.julian_day(date)
    assert np.isclose(v, expected_value)


@pytest.mark.parametrize(
    "date,expected_value",
    [
        (datetime.datetime(2024, 4, 22), (12.235799080498582, 0.40707190497656276)),
        (
            datetime.datetime(2024, 4, 22, 12, 0, 0),
            (12.403019177270453, 0.43253901867797273),
        ),
    ],
)
def test_solar_declination_angle(date, expected_value):
    declination, time_correction = solar.solar_declination_angle(date)
    assert np.isclose(declination, expected_value[0])
    assert np.isclose(time_correction, expected_value[1])


def test_cos_solar_zenith_angle():
    date = datetime.datetime(2024, 4, 22, 12, 0, 0)
    latitudes = np.array([40.0])
    longitudes = np.array([18.0])

    v = solar.cos_solar_zenith_angle(date, latitudes, longitudes)
    assert np.isclose(v[0], 0.8478445449796352)


def test_cos_solar_zenith_angle_integrated():
    begin_date = datetime.datetime(2024, 4, 22)
    end_date = datetime.datetime(2024, 4, 23)
    latitudes = np.array([40.0])
    longitudes = np.array([18.0])

    v = solar.cos_solar_zenith_angle_integrated(begin_date, end_date, latitudes, longitudes)
    assert np.isclose(v[0], 0.3110738757)


def test_incoming_solar_radiation():
    date = datetime.datetime(2024, 4, 22, 12, 0, 0)
    v = solar.incoming_solar_radiation(date)
    assert np.isclose(v, 4833557.3088814365)


def test_toa_incident_solar_radiation():
    begin_date = datetime.datetime(2024, 4, 22)
    end_date = datetime.datetime(2024, 4, 23)
    latitudes = np.array([40.0])
    longitudes = np.array([18.0])

    v = solar.toa_incident_solar_radiation(begin_date, end_date, latitudes, longitudes)
    assert np.isclose(v, 1503617.8237746414)
