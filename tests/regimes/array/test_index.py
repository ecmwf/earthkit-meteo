# (C) Copyright 2025- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import numpy as np
import pytest

from earthkit.meteo.regimes import array


@pytest.fixture
def patterns():
    class MockPatterns:
        _lat = np.linspace(90.0, 0.0, 91)
        _lon = np.linspace(60.0, -60.0, 121)
        _dipole = np.cos(np.deg2rad(_lon[None, :])) * np.cos(np.deg2rad(_lat[:, None]) * 2)
        _monopole = np.cos(np.deg2rad(_lon[None, :])) * np.sin(np.deg2rad(_lat[:, None]) * 2)
        shape = (91, 121)
        grid = {
            "grid": [1.0, 1.0],
            "area": [max(_lat), min(_lon), min(_lat), max(_lon)],
        }

        def patterns(self, multiple=False):
            out = np.asarray([self._dipole, self._monopole, -self._dipole])
            if multiple:
                out = np.stack([out, 2 * out])
            return out

    return MockPatterns()


def test_project_matches_field_and_pattern_shapes(patterns):
    with pytest.raises(ValueError):
        array.project(np.zeros((91 * 121,)), patterns, weights=None)
    with pytest.raises(ValueError):
        array.project(np.zeros((20, 30)), patterns, weights=None)
    with pytest.raises(ValueError):
        array.project(np.zeros((91, 2, 3)), patterns, weights=None)


def test_project_matches_weights_and_pattern_shapes(patterns):
    with pytest.raises(ValueError):
        array.project(np.ones(patterns.shape), patterns, weights=np.ones((20, 30)))


def test_project_ones_with_uniform_weights(patterns):
    result = array.project(np.ones(patterns.shape), patterns, weights=np.ones(patterns.shape))
    reference = [np.mean(patterns._dipole), np.mean(patterns._monopole), np.mean(-patterns._dipole)]
    np.testing.assert_allclose(result, reference)


def test_project_ones_with_coslat_weights(patterns):
    lat_2d = np.repeat(patterns._lat, patterns._lon.size).reshape(patterns.shape)
    coslat = np.cos(np.deg2rad(lat_2d))
    proj = array.project(np.ones(patterns.shape), patterns, weights=coslat)
    assert proj.shape == (3,)
    # Dipole plausibility
    assert proj[0] > 0  # positive values where weights are heigher
    assert proj[2] < 0  # negative values where weights are higher
    np.testing.assert_allclose(proj[0], -proj[2])


def test_project_zeros_returns_zero(patterns):
    proj = array.project(np.zeros(patterns.shape), patterns, weights=np.ones(patterns.shape))
    np.testing.assert_allclose(proj, 0.0)


def test_project_is_commutative(patterns):
    fields = np.stack([patterns._dipole, patterns._monopole])
    proj = array.project(fields, patterns, weights=np.ones(patterns.shape))
    np.testing.assert_allclose(proj[0, 1], proj[1, 0])


def test_project_maintains_shape(patterns):
    fields = np.zeros((2, 3, 4, *patterns.shape))
    proj = array.project(fields, patterns, weights=np.ones(patterns.shape))
    assert proj.shape == (2, 3, 4, 3)


@pytest.mark.xfail(reason="grid info not available from earthkit-geo")
def test_project_generates_weights_by_default(patterns):
    array.project(np.ones(patterns.shape), patterns)


def test_project_with_single_pattern_return(patterns):
    proj = array.project(
        np.ones((2, *patterns.shape)), patterns, weights=np.ones(patterns.shape), patterns_coords={"multiple": False}
    )
    # All patterns are the same; monopole has nonzero projection
    assert proj.shape == (2, 3)
    np.testing.assert_allclose(proj[0, 1], proj[1, 1])


def test_project_with_multiple_pattern_return(patterns):
    proj = array.project(
        np.ones((2, *patterns.shape)), patterns, weights=np.ones(patterns.shape), patterns_coords={"multiple": True}
    )
    # Second pattern has twice the amplitude; monopole has nonzero projection
    assert proj.shape == (2, 3)
    np.testing.assert_allclose(proj[0, 1], 0.5 * proj[1, 1])


def test_regime_index():
    proj = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
    mean = np.asarray([2.0, -4.0])
    std = np.asarray([10.0, 2.0])

    result = array.regime_index(proj, mean, std)
    reference = np.asarray([[-0.2, 2.0], [-0.1, 2.5], [0.0, 3.0], [0.1, 3.5], [0.2, 4.0], [0.3, 4.5]])

    assert result.shape == proj.shape
    np.testing.assert_allclose(result, reference)
