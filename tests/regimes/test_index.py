# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

from earthkit.meteo import regimes


@pytest.fixture
def patterns():
    class MockRegimePatterns:

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

    return MockRegimePatterns()


def test_project_matches_field_and_pattern_shapes(patterns):
    with pytest.raises(AssertionError):
        regimes.project(np.zeros((91 * 121,)), patterns)
    with pytest.raises(AssertionError):
        regimes.project(np.zeros((20, 30)), patterns)
    with pytest.raises(AssertionError):
        regimes.project(np.zeros((91, 2, 3)), patterns)


def test_project_ones_with_uniform_weights(patterns):
    proj = regimes.project(np.ones(patterns.shape), patterns, weights=np.ones(patterns.shape))
    assert np.isclose(proj["dipole"], np.mean(patterns._dipole))
    assert np.isclose(proj["monopole"], np.mean(patterns._monopole))
    assert np.isclose(proj["dipole_inv"], np.mean(patterns._dipole))
    # Pattern symmetry
    assert np.isclose(proj["dipole"], -proj["dipole_inv"])


def test_project_ones_with_coslat_weights(patterns):
    lat_2d = np.repeat(patterns._lat, patterns._lon.size).reshape(patterns.shape)
    coslat = np.cos(np.deg2rad(lat_2d))
    proj = regimes.project(np.ones(patterns.shape), patterns, weights=coslat)
    assert proj["dipole"] > 0  # positive values where weights are heigher
    assert proj["dipole_inv"] < 0  # negative values where weights are higher
    assert np.isclose(proj["dipole"], -proj["dipole_inv"])


def test_project_zeros_returns_zero(patterns):
    proj = regimes.project(np.zeros(patterns.shape), patterns, weights=np.ones(patterns.shape))
    assert np.isclose(proj["dipole"], 0.0)
    assert np.isclose(proj["monopole"], 0.0)
    assert np.isclose(proj["dipole_inv"], 0.0)


def test_project_is_commutative(patterns):
    fields = np.stack([patterns._dipole, patterns._monopole])
    proj = regimes.project(fields, patterns, weights=np.ones(patterns.shape))
    np.testing.assert_allclose(proj["dipole"][1], proj["monopole"][0])


def test_project_maintains_shape(patterns):
    fields = np.zeros((2, 3, 4, *patterns.shape))
    proj = regimes.project(fields, patterns, weights=np.ones(patterns.shape))
    assert proj["dipole"].shape == (2, 3, 4)


@pytest.mark.xfail(reason="grid info not available from earthkit-geo")
def test_project_generates_weights_by_default(patterns):
    regimes.project(np.ones(patterns.shape), patterns)


def test_project_with_single_pattern_return(patterns):
    proj = regimes.project(
        np.ones((2, *patterns.shape)), patterns, weights=np.ones(patterns.shape), single=True
    )
    # All patterns are the same
    assert proj["dipole"].shape == (2,)
    assert np.isclose(proj["dipole"][0], proj["dipole"][1])


def test_project_with_multiple_pattern_return(patterns):
    proj = regimes.project(
        np.ones((2, *patterns.shape)), patterns, weights=np.ones(patterns.shape), single=False
    )
    # Second pattern has twice the amplitude
    assert proj["dipole"].shape == (2,)
    assert np.isclose(proj["dipole"][0], 0.5 * proj["dipole"][1])


def test_standardise_with_dict():
    proj = {
        "foo": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        "bar": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    mean = {"foo": 2.0, "bar": -4.0}
    std = {"foo": 10.0, "bar": 2.0}

    index = regimes.standardise(proj, mean, std)

    assert len(index) == 2
    assert "foo" in index
    assert "bar" in index
    np.testing.assert_allclose(index["foo"], [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(index["bar"], [2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
