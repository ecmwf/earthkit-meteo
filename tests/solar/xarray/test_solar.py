# (C) Copyright 2026 ECMWF.
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

import earthkit.meteo.solar.array as array_solar
from earthkit.meteo import solar
from earthkit.meteo.utils.testing import NO_XARRAY

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})

pytestmark = pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")


def _da(values):
    import xarray as xr

    return xr.DataArray(np.asarray(values))


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_julian_day():
    dates = [
        datetime.datetime(2024, 4, 22, 0, 0, 0),
        datetime.datetime(2024, 4, 22, 12, 0, 0),
    ]

    da = _da(dates)
    assert np.issubdtype(da.dtype, np.datetime64)

    v = solar.julian_day(da)
    assert hasattr(v, "values")
    assert np.allclose(v.values, np.asarray([112.0, 112.5]))


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_solar_declination_angle():
    dates = [
        datetime.datetime(2024, 4, 22, 0, 0, 0),
        datetime.datetime(2024, 4, 22, 12, 0, 0),
    ]

    da = _da(dates)
    assert np.issubdtype(da.dtype, np.datetime64)

    decl, tc = solar.solar_declination_angle(da)

    decl_ref = np.asarray([12.235799080498582, 12.403019177270453])
    tc_ref = np.asarray([0.40707190497656276, 0.43253901867797273])

    assert np.allclose(decl.values, decl_ref)
    assert np.allclose(tc.values, tc_ref)


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_incoming_solar_radiation():
    dates = [datetime.datetime(2024, 4, 22, 12, 0, 0)]

    da = _da(dates)
    assert np.issubdtype(da.dtype, np.datetime64)

    v = solar.incoming_solar_radiation(da)
    assert np.allclose(v.values, np.asarray([4833557.3088814365]))


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_cos_solar_zenith_angle():
    import xarray as xr

    date = datetime.datetime(2024, 4, 22, 12, 0, 0)
    lat = xr.DataArray(np.asarray([40.0, 41.0]), dims=("point",))
    lon = xr.DataArray(np.asarray([18.0, 19.0]), dims=("point",))

    v = solar.cos_solar_zenith_angle(date, lat, lon)

    v_ref = array_solar.cos_solar_zenith_angle(date, lat.values, lon.values)
    assert np.allclose(v.values, np.asarray(v_ref))


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize("integration_order", [1, 2, 3, 4])
def test_xr_cos_solar_zenith_angle_integrated(integration_order):
    import xarray as xr

    begin_date = datetime.datetime(2024, 4, 22)
    end_date = datetime.datetime(2024, 4, 23)
    lat = xr.DataArray(np.asarray([40.0]), dims=("point",))
    lon = xr.DataArray(np.asarray([18.0]), dims=("point",))

    v = solar.cos_solar_zenith_angle_integrated(
        begin_date,
        end_date,
        lat,
        lon,
        integration_order=integration_order,
    )

    v_ref = array_solar.cos_solar_zenith_angle_integrated(
        begin_date,
        end_date,
        lat.values,
        lon.values,
        integration_order=integration_order,
    )

    assert np.allclose(v.values, np.asarray(v_ref))


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_toa_incident_solar_radiation():
    import xarray as xr

    begin_date = datetime.datetime(2024, 4, 22)
    end_date = datetime.datetime(2024, 4, 23)
    lat = xr.DataArray(np.asarray([40.0]), dims=("point",))
    lon = xr.DataArray(np.asarray([18.0]), dims=("point",))

    v = solar.toa_incident_solar_radiation(begin_date, end_date, lat, lon)

    v_ref = array_solar.toa_incident_solar_radiation(
        begin_date,
        end_date,
        lat.values,
        lon.values,
    )

    assert np.allclose(v.values, np.asarray(v_ref))
