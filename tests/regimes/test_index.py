# (C) Copyright 2025- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import numpy as np
import pytest

from earthkit.meteo import regimes
from earthkit.meteo.regimes import array as regimes_array
from earthkit.meteo.regimes import xarray as regimes_xarray


@pytest.fixture
def patterns():
    return regimes.ConstantPatterns(
        labels=["foo", "bar"],
        patterns=[[[1.0, 1.0]], [[0.1, 0.9]]],
        grid={"grid": [1.0, 1.0], "area": [45.0, 0.0, 45.0, 1.0]},
    )


def test_highlevel_project_dispatches_to_array(patterns):
    field = np.ones(patterns.shape)
    weights = np.ones(patterns.shape)

    got = regimes.project(field, patterns, weights)
    ref = regimes_array.project(field, patterns, weights)

    assert isinstance(got, np.ndarray)
    np.testing.assert_allclose(got, ref)


def test_highlevel_regime_index_dispatches_to_array():
    projections = np.asarray([[0.0, 1.0]])
    mean = np.asarray([0.5])
    std = np.asarray([0.5])

    got = regimes.regime_index(projections, mean, std)
    ref = regimes_array.regime_index(projections, mean, std)

    assert isinstance(got, np.ndarray)
    np.testing.assert_allclose(got, ref)


xr = pytest.importorskip("xarray")


@pytest.fixture
def xarray_data3d():
    data2d = xr.DataArray(
        data=[[0.0, 1.0]],
        coords={
            "lat": (["lat"], [45.0]),
            "lon": (["lon"], [0.0, 1.0]),
        },
        dims=["lat", "lon"],
    )
    return data2d.expand_dims({"foo": [1.0, 2.0, 4.0]})


@pytest.fixture
def xarray_weights1d():
    return xr.DataArray(data=[1.0, 3.0], coords={"lat": (["lat"], [60.0, 50.0])}, dims=["lat"])


def test_highlevel_project_dispatches_to_xarray(xarray_data3d, patterns, xarray_weights1d):
    got = regimes.project(xarray_data3d, patterns, xarray_weights1d)
    ref = regimes_xarray.project(xarray_data3d, patterns, xarray_weights1d)

    xr.testing.assert_allclose(got, ref)


def test_highlevel_regime_index_dispatches_to_xarray(xarray_data3d):
    mean = xarray_data3d.mean(dim=["lat", "lon"])
    std = xarray_data3d.std(dim=["lat", "lon"])

    got = regimes.regime_index(xarray_data3d, mean, std)
    ref = regimes_xarray.regime_index(xarray_data3d, mean, std)

    xr.testing.assert_allclose(got, ref)
