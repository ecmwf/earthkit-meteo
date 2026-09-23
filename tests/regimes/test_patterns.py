# (C) Copyright 2025- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import numpy as np
import pytest

from earthkit.meteo import regimes


class TestConstantPatterns:
    lat = np.linspace(90.0, 0.0, 91)
    lon = np.linspace(60.0, -60.0, 121)
    dipole = np.cos(np.deg2rad(lon[None, :])) * np.cos(np.deg2rad(lat[:, None]) * 2)
    monopole = np.cos(np.deg2rad(lon[None, :])) * np.sin(np.deg2rad(lat[:, None]) * 2)

    @pytest.fixture
    def patterns(self):
        return regimes.ConstantPatterns(
            labels=["dipole", "monopole", "dipole_inv"],
            patterns=np.stack([self.dipole, self.monopole, -self.dipole]).copy(),
        )

    zeros = np.zeros(shape=(lat.size, lon.size))
    ones = np.ones(shape=(lat.size, lon.size))

    def test_shape(self, patterns):
        assert patterns.shape == (self.lat.size, self.lon.size)

    def test_size(self, patterns):
        assert patterns.size == self.lat.size * self.lon.size

    def test_ndim(self, patterns):
        assert patterns.ndim == 2

    def test_len(self, patterns):
        assert len(patterns) == 3
        assert len(patterns) == len(patterns.labels)

    def test_patterns(self, patterns):
        pat = patterns.patterns()
        assert pat.shape == (3, 91, 121)
        np.testing.assert_allclose(pat[0], self.dipole)
        np.testing.assert_allclose(pat[1], self.monopole)


class TestModulatedPatterns:
    lat = np.linspace(90.0, 0.0, 91)
    lon = np.linspace(60.0, -60.0, 121)
    dipole = np.cos(np.deg2rad(lon[None, :])) * np.cos(np.deg2rad(lat[:, None]) * 2)

    @pytest.fixture
    def patterns(self):
        return regimes.ModulatedPatterns(
            labels=["dipole"],
            base_patterns=np.stack([self.dipole]).copy(),
            modulator=lambda x, y: y * np.sign(x),
        )

    @pytest.fixture
    def data_xr(self):
        xr = pytest.importorskip("xarray")

        return xr.DataArray(
            data=np.ones((4, 3, 2, self.lat.size, self.lon.size)),
            coords={
                "foo": (["foo"], [1.0, 2.0, 3.0, 4.0]),
                "bar": (["bar"], [-1.0, 1.0, 2.0]),
                "baz": (["baz"], [3.0, -1.0]),
                "lat": (["lat"], self.lat),
                "lon": (["lon"], self.lon),
            },
            dims=["foo", "bar", "baz", "lat", "lon"],
        )

    def test_shape(self, patterns):
        assert patterns.shape == (self.lat.size, self.lon.size)

    def test_size(self, patterns):
        assert patterns.size == self.lat.size * self.lon.size

    def test_ndim(self, patterns):
        assert patterns.ndim == 2

    def test_patterns_one_argument_scalar(self, patterns):
        pat = patterns.patterns(x=[3.0, 0.0, -4.0], y=1.0)
        assert pat.shape == (3, 1, *self.dipole.shape)
        np.testing.assert_allclose(pat[0, 0], self.dipole)
        np.testing.assert_allclose(pat[1, 0], 0.0)
        np.testing.assert_allclose(pat[2, 0], -self.dipole)

    def test_patterns_both_arguments_vectors(self, patterns):
        pat = patterns.patterns(x=[3.0, -4.0], y=[1.0, 2.0])
        assert pat.shape == (2, 1, *self.dipole.shape)
        np.testing.assert_allclose(pat[0, 0], self.dipole)
        np.testing.assert_allclose(pat[1, 0], -2 * self.dipole)


class TestPatternsWithGrid:
    @pytest.fixture
    def TestPatterns(self):
        class TestPatterns(regimes.Patterns):
            def __init__(self, **kwargs):
                super().__init__(["a", "b"], **kwargs)

            def patterns(self):
                pass

        return TestPatterns

    @pytest.fixture
    def Grid(self):
        ekg = pytest.importorskip("earthkit.geo")
        return ekg.grids.Grid

    def test_shape_or_grid_is_required(self, TestPatterns):
        with pytest.raises(ValueError):
            TestPatterns(shape=None, grid=None)

    def test_explicit_shape_is_used_without_grid(self, TestPatterns):
        patterns = TestPatterns(shape=(3, 4), grid=None)
        assert patterns.shape == (3, 4)
        assert patterns.size == 12
        assert patterns.ndim == 2

    def test_shape_is_inferred_from_grid_if_absent(self, TestPatterns, Grid):
        grid = Grid("O16")
        patterns = TestPatterns(shape=None, grid=grid)
        assert patterns.shape == grid.shape

    @pytest.mark.parametrize(
        "grid_spec",
        [
            "O16",
            "N16",
            {"grid": [5.0, 2.5]},
            {"grid": [0.5, 0.5], "area": [90, -80, 30, 40]},
        ],
    )
    def test_shape_and_grid_shape_must_match_if_both_given(self, TestPatterns, Grid, grid_spec):
        grid = Grid(grid_spec)
        TestPatterns(shape=grid.shape, grid=grid)
        with pytest.raises(ValueError):
            TestPatterns(shape=(1, 1, 2, 3, 4, 5), grid=grid)
