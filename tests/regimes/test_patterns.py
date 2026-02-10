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
            modulator=lambda x: np.sign(x),  # clarify the arg name
        )

    def test_shape(self, patterns):
        assert patterns.shape == (self.lat.size, self.lon.size)

    def test_size(self, patterns):
        assert patterns.size == self.lat.size * self.lon.size

    def test_ndim(self, patterns):
        assert patterns.ndim == 2

    def test_patterns(self, patterns):
        pat = patterns.patterns(x=[3.0, 0.0, -4.0])
        assert len(pat) == 1
        np.testing.assert_allclose(pat["dipole"][0], self.dipole)
        np.testing.assert_allclose(pat["dipole"][1], 0.0)
        np.testing.assert_allclose(pat["dipole"][2], -self.dipole)
