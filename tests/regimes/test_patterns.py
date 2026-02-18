# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

from earthkit.meteo import regimes

# from earthkit.meteo.utils.testing import NO_XARRAY


class TestConstantPatterns:

    lat = np.linspace(90.0, 0.0, 91)
    lon = np.linspace(60.0, -60.0, 121)
    dipole = np.cos(np.deg2rad(lon[None, :])) * np.cos(np.deg2rad(lat[:, None]) * 2)
    monopole = np.cos(np.deg2rad(lon[None, :])) * np.sin(np.deg2rad(lat[:, None]) * 2)

    @pytest.fixture
    def patterns(self):
        return regimes.ConstantPatterns(
            labels=["dipole", "monopole", "dipole_inv"],
            grid={"grid": [1.0, 1.0], "area": [max(self.lat), min(self.lon), min(self.lat), max(self.lon)]},
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

    def test_patterns(self, patterns):
        pat = patterns.patterns()
        assert len(pat) == 3
        np.testing.assert_allclose(pat["dipole"], self.dipole)
        np.testing.assert_allclose(pat["monopole"], self.monopole)


class TestModulatedPatterns:

    lat = np.linspace(90.0, 0.0, 91)
    lon = np.linspace(60.0, -60.0, 121)
    dipole = np.cos(np.deg2rad(lon[None, :])) * np.cos(np.deg2rad(lat[:, None]) * 2)

    @pytest.fixture
    def patterns(self):
        return regimes.ModulatedPatterns(
            labels=["dipole"],
            grid={"grid": [1.0, 1.0], "area": [max(self.lat), min(self.lon), min(self.lat), max(self.lon)]},
            patterns=np.stack([self.dipole]).copy(),
            modulator=lambda x, y: y * np.sign(x),
        )

    @pytest.fixture
    def data_xr(self):
        import xarray as xr

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
        assert len(pat) == 1
        assert pat["dipole"].shape == (3, *self.dipole.shape)
        np.testing.assert_allclose(pat["dipole"][0], self.dipole)
        np.testing.assert_allclose(pat["dipole"][1], 0.0)
        np.testing.assert_allclose(pat["dipole"][2], -self.dipole)

    def test_patterns_both_arguments_vectors(self, patterns):
        pat = patterns.patterns(x=[3.0, -4.0], y=[1.0, 2.0])
        assert len(pat) == 1
        assert pat["dipole"].shape == (2, *self.dipole.shape)
        np.testing.assert_allclose(pat["dipole"][0], self.dipole)
        np.testing.assert_allclose(pat["dipole"][1], -2 * self.dipole)

    # @pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
    def test_patterns_iterxr_with_extra_coordinate_mapping(self, data_xr, patterns):
        it = patterns._patterns_iterxr(data_xr, patterns_extra_coords={"x": "baz", "y": "bar"})
        name, pat = next(it)
        assert name == "dipole"
        assert pat.dims == ("bar", "baz", "lat", "lon")
        # In contrast to array-level .patterns, _patterns_iterxr takes cartesian product
        assert pat.shape == (3, 2, self.lat.size, self.lon.size)
        np.testing.assert_allclose(pat.sel(bar=-1.0, baz=3.0).values, -self.dipole)
        np.testing.assert_allclose(pat.sel(bar=-1.0, baz=-1).values, self.dipole)
        np.testing.assert_allclose(pat.sel(bar=1.0, baz=3.0).values, self.dipole)
        np.testing.assert_allclose(pat.sel(bar=1.0, baz=-1).values, -self.dipole)
        np.testing.assert_allclose(pat.sel(bar=2.0, baz=3.0).values, 2.0 * self.dipole)
        np.testing.assert_allclose(pat.sel(bar=2.0, baz=-1).values, -2.0 * self.dipole)
        with pytest.raises(StopIteration):
            next(it)
