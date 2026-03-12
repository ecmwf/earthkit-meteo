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


class _ArrayPatterns:
    _lat = np.linspace(90.0, 0.0, 91)
    _lon = np.linspace(60.0, -60.0, 121)
    _dipole = np.cos(np.deg2rad(_lon[None, :])) * np.cos(np.deg2rad(_lat[:, None]) * 2)
    _monopole = np.cos(np.deg2rad(_lon[None, :])) * np.sin(np.deg2rad(_lat[:, None]) * 2)
    shape = (91, 121)
    grid = {"grid": [1.0, 1.0], "area": [max(_lat), min(_lon), min(_lat), max(_lon)]}

    def patterns(self, single=True):
        return {
            "dipole": self._dipole if single else np.stack([self._dipole, 2 * self._dipole]),
            "monopole": self._monopole if single else np.stack([self._monopole, 2 * self._monopole]),
            "dipole_inv": -self._dipole if single else np.stack([-self._dipole, -2 * self._dipole]),
        }


@pytest.fixture
def array_patterns():
    return _ArrayPatterns()


def test_highlevel_project_dispatches_to_array(array_patterns):
    field = np.ones(array_patterns.shape)
    weights = np.ones(array_patterns.shape)

    got = regimes.project(field, array_patterns, weights)
    ref = regimes_array.project(field, array_patterns, weights)

    assert isinstance(got, dict)
    assert got.keys() == ref.keys()
    for key in got:
        np.testing.assert_allclose(got[key], ref[key])


def test_highlevel_regime_index_array_like_is_rejected():
    projections = {"foo": np.asarray([0.0, 1.0])}
    mean = {"foo": 0.5}
    std = {"foo": 0.5}

    with pytest.raises(TypeError, match="No matching dispatcher found"):
        regimes.regime_index(projections, mean, std)


xr = pytest.importorskip("xarray")


class _XarrayPatterns:
    shape = (2, 4)
    size = 2 * 4
    ndim = 2

    def _patterns_iterxr(self, reference_da, patterns_extra_coords):
        foo = reference_da["foo"].values
        a = foo[:, None, None] * np.asarray([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
        b = foo[:, None, None] * np.asarray([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        coords = {"foo": reference_da["foo"], "lat": reference_da["lat"], "lon": reference_da["lon"]}
        dims = ["foo", "lat", "lon"]
        yield "a", xr.DataArray(a, coords, dims)
        yield "b", xr.DataArray(b, coords, dims)


@pytest.fixture
def xarray_patterns():
    return _XarrayPatterns()


@pytest.fixture
def xarray_data3d():
    data2d = xr.DataArray(
        data=[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
        coords={"lat": (["lat"], [60.0, 50.0]), "lon": (["lon"], [-10.0, 0.0, 10.0, 20.0])},
        dims=["lat", "lon"],
    )
    return data2d.expand_dims({"foo": [1.0, 2.0, 4.0]})


@pytest.fixture
def xarray_weights1d():
    return xr.DataArray(data=[1.0, 3.0], coords={"lat": (["lat"], [60.0, 50.0])}, dims=["lat"])


def test_highlevel_project_dispatches_to_xarray(xarray_data3d, xarray_patterns, xarray_weights1d):
    got = regimes.project(xarray_data3d, xarray_patterns, xarray_weights1d)
    ref = regimes_xarray.project(xarray_data3d, xarray_patterns, xarray_weights1d)

    xr.testing.assert_allclose(got, ref)


def test_highlevel_regime_index_dispatches_to_xarray(xarray_data3d):
    mean = xarray_data3d.mean(dim=["lat", "lon"])
    std = xarray_data3d.std(dim=["lat", "lon"])

    got = regimes.regime_index(xarray_data3d, mean, std)
    ref = regimes_xarray.regime_index(xarray_data3d, mean, std)

    xr.testing.assert_allclose(got, ref)
