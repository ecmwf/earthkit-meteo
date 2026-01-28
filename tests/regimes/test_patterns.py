# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from earthkit.meteo import regimes


class TestConstantRegimePatterns:

    lat = np.linspace(90.0, 0.0, 91)
    lon = np.linspace(60.0, -60.0, 121)
    dipole = np.cos(np.deg2rad(lon[None, :])) * np.cos(np.deg2rad(lat[:, None]) * 2)
    monopole = np.cos(np.deg2rad(lon[None, :])) * np.sin(np.deg2rad(lat[:, None]) * 2)
    patterns = regimes.ConstantRegimePatterns(
        regimes=["dipole", "monopole", "dipole_inv"],
        grid={"grid": [1.0, 1.0], "area": [max(lat), min(lon), min(lat), max(lon)]},
        patterns=np.stack([dipole, monopole, -dipole]),
    )

    zeros = np.zeros(shape=patterns.shape)
    ones = np.ones(shape=patterns.shape)

    def test_shape(self):
        assert self.patterns.shape == (self.lat.size, self.lon.size)

    def test_patterns(self):
        pat = self.patterns.patterns()
        assert len(pat) == 3
        np.testing.assert_allclose(pat["dipole"], self.dipole)
        np.testing.assert_allclose(pat["monopole"], self.monopole)


class TestModulatedRegimePatterns:

    lat = np.linspace(90.0, 0.0, 91)
    lon = np.linspace(60.0, -60.0, 121)
    dipole = np.cos(np.deg2rad(lon[None, :])) * np.cos(np.deg2rad(lat[:, None]) * 2)
    patterns = regimes.ModulatedRegimePatterns(
        regimes=["dipole"],
        grid={"grid": [1.0, 1.0], "area": [max(lat), min(lon), min(lat), max(lon)]},
        patterns=np.stack([dipole]),
        modulator=lambda x: np.sign(x),  # clarify the arg name
    )

    def test_shape(self):
        assert self.patterns.shape == (self.lat.size, self.lon.size)

    def test_patterns(self):
        pat = self.patterns.patterns(x=[3.0, 0.0, -4.0])
        assert len(pat) == 1
        np.testing.assert_allclose(pat["dipole"][0], self.dipole)
        np.testing.assert_allclose(pat["dipole"][1], 0.0)
        np.testing.assert_allclose(pat["dipole"][2], -self.dipole)
