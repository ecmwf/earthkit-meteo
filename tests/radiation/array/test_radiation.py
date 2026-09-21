# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Tests for the array level radiation functions."""

import numpy as np
import pytest
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo import constants
from earthkit.meteo.radiation import array as radiation

DIFFUSE = [0.0, 120.5, 310.0]  # W/m2
DIRECT = [0.0, 45.25, 590.0]  # W/m2
TOTAL = [0.0, 165.75, 900.0]  # W/m2


@pytest.mark.parametrize("xp,device", NAMESPACE_DEVICES)
def test_downward_shortwave_radiation(xp, device):
    diffuse = xp.asarray(DIFFUSE, device=device)
    direct = xp.asarray(DIRECT, device=device)

    out = radiation.surface_downward_shortwave_radiation(diffuse, direct)

    assert out.shape == diffuse.shape
    np.testing.assert_allclose(np.asarray(out), np.array(TOTAL))


@pytest.mark.parametrize("xp,device", NAMESPACE_DEVICES)
def test_downward_shortwave_radiation_broadcast(xp, device):
    """A scalar component is broadcast against a gridded one."""
    diffuse = xp.asarray(DIFFUSE, device=device)

    out = radiation.surface_downward_shortwave_radiation(diffuse, 10.0)

    np.testing.assert_allclose(np.asarray(out), np.array(DIFFUSE) + 10.0)


def test_downward_shortwave_radiation_list_input():
    """Lists are converted to arrays rather than concatenated."""
    out = radiation.surface_downward_shortwave_radiation(DIFFUSE, DIRECT)

    np.testing.assert_allclose(np.asarray(out), np.array(TOTAL))


def test_downward_shortwave_radiation_nan_propagates():
    diffuse = np.array([1.0, np.nan])
    direct = np.array([2.0, 3.0])

    out = radiation.surface_downward_shortwave_radiation(diffuse, direct)

    np.testing.assert_allclose(out, np.array([3.0, np.nan]))


NET_LONGWAVE = [-60.0, -30.0, -105.5]  # W/m2
SURFACE_TEMPERATURE = [280.0, 300.0, 265.3]  # K


@pytest.mark.parametrize("xp,device", NAMESPACE_DEVICES)
def test_surface_downward_longwave_flux(xp, device):
    net_longwave = xp.asarray(NET_LONGWAVE, device=device)
    surface_temperature = xp.asarray(SURFACE_TEMPERATURE, device=device)

    out = radiation.surface_downward_longwave_flux(net_longwave, surface_temperature)

    ref = np.array(NET_LONGWAVE) / constants.emissivity_surface + constants.sigma * np.array(SURFACE_TEMPERATURE) ** 4
    np.testing.assert_allclose(np.asarray(out), ref)


def test_surface_downward_longwave_flux_emissivity():
    """A unit emissivity reduces the budget to the Stefan-Boltzmann term."""
    surface_temperature = np.array(SURFACE_TEMPERATURE)

    out = radiation.surface_downward_longwave_flux(0.0, surface_temperature, emissivity=1.0)

    np.testing.assert_allclose(out, constants.sigma * surface_temperature**4)


def test_surface_downward_longwave_flux_roundtrip():
    """The downward flux inverts the net flux of a grey body at the same temperature."""
    surface_temperature = np.array(SURFACE_TEMPERATURE)
    downward = np.array([300.0, 400.0, 250.0])
    emissivity = 0.98
    net_longwave = emissivity * (downward - constants.sigma * surface_temperature**4)

    out = radiation.surface_downward_longwave_flux(net_longwave, surface_temperature, emissivity=emissivity)

    np.testing.assert_allclose(out, downward)


@pytest.mark.parametrize("xp,device", NAMESPACE_DEVICES)
def test_downward_shortwave_radiation_clipped(xp, device):
    """Negative sums are clipped, NaNs are not."""
    diffuse = xp.asarray([-5.0, 1.0, np.nan], device=device)
    direct = xp.asarray([2.0, -4.0, 1.0], device=device)

    out = radiation.surface_downward_shortwave_radiation(diffuse, direct)

    np.testing.assert_allclose(np.asarray(out), np.array([0.0, 0.0, np.nan]))
