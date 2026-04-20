# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest

import earthkit.meteo.wind.array as array_wind
from earthkit.meteo.utils.testing import IN_GITHUB, NO_EKD

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


@pytest.mark.skipif(IN_GITHUB, reason="Disabled in github CI")
@pytest.mark.skipif(NO_EKD, reason="earthkit.data is not installed")
def test_fieldlist_wind_speed():
    import earthkit.data as ekd

    import earthkit.meteo.wind.fieldlist as wind

    ds = ekd.from_source("sample", "tuv_pl.grib").to_fieldlist()

    u = ds.sel({"parameter.variable": "u"}).order_by("level")
    v = ds.sel({"parameter.variable": "v"}).order_by("level")
    res = wind.speed(u, v)

    assert len(u) == 6
    assert len(res) == 6
    assert res.get("parameter.variable") == ["ws"] * 6
    assert res.values.shape == u.values.shape

    ref = array_wind.speed(u[0].values, v[0].values)
    assert np.allclose(res[0].values, ref, equal_nan=True)


@pytest.mark.skipif(IN_GITHUB, reason="Disabled in github CI")
@pytest.mark.skipif(NO_EKD, reason="earthkit.data is not installed")
def test_fieldlist_wind_direction():
    import earthkit.data as ekd

    import earthkit.meteo.wind.fieldlist as wind

    ds = ekd.from_source("sample", "tuv_pl.grib").to_fieldlist()

    u = ds.sel({"parameter.variable": "u"}).order_by("level")
    v = ds.sel({"parameter.variable": "v"}).order_by("level")
    res = wind.direction(u, v)

    assert len(u) == 6
    assert len(res) == 6
    assert res.get("parameter.variable") == ["wdir"] * 6
    assert res.values.shape == u.values.shape

    ref = array_wind.direction(u[0].values, v[0].values)
    assert np.allclose(res[0].values, ref, equal_nan=True)


@pytest.mark.skipif(IN_GITHUB, reason="Disabled in github CI")
@pytest.mark.skipif(NO_EKD, reason="earthkit.data is not installed")
def test_fieldlist_w_from_omega():
    import earthkit.data as ekd

    import earthkit.meteo.wind.fieldlist as wind

    ds = ekd.from_source("sample", "omega_pl.grib").to_fieldlist()

    omega = ds.sel({"parameter.variable": "w"}).order_by("level")
    t = ds.sel({"parameter.variable": "t"}).order_by("level")
    res = wind.w_from_omega(omega, t, p=None)

    assert len(omega) == 2
    assert len(res) == 2
    assert res.get("parameter.variable") == ["wz"] * 2
    assert res.values.shape == omega.values.shape

    ref = array_wind.w_from_omega(omega[0].values, t[0].values, omega[0].metadata("level") * 100.0)
    assert np.allclose(res[0].values, ref, equal_nan=True)


@pytest.mark.skipif(IN_GITHUB, reason="Disabled in github CI")
@pytest.mark.skipif(NO_EKD, reason="earthkit.data is not installed")
def test_fieldlist_coriolis():
    import earthkit.data as ekd

    import earthkit.meteo.wind.fieldlist as wind

    ds = ekd.from_source("sample", "tuv_pl.grib").to_fieldlist()

    res = wind.coriolis(ds)

    assert len(res) == 18
    assert res.get("parameter.variable") == ["fc"] * 18
    assert res.values.shape == ds.values.shape

    ref = array_wind.coriolis(ds[0].geography.latitudes().reshape(ds[0].values.shape))
    assert np.allclose(res[0].values, ref, equal_nan=True)
