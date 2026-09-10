# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest

from earthkit.meteo import radiation
from earthkit.meteo.utils.testing import NO_XARRAY

pytestmark = pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")

DIFFUSE = [0.0, 120.5, 310.0]  # W/m2
DIRECT = [0.0, 45.25, 590.0]  # W/m2


def _da(x, dims=None):
    import xarray as xr

    values = np.asarray(x)
    return xr.DataArray(values, dims=dims) if dims else xr.DataArray(values)


@pytest.mark.parametrize(
    "diffuse,direct",
    [
        (DIFFUSE, DIRECT),
        (120.5, 45.25),
    ],
)
def test_xr_downward_shortwave_radiation(diffuse, direct):
    out = radiation.surface_downward_shortwave_radiation(_da(diffuse), _da(direct))
    ref = radiation.array.surface_downward_shortwave_radiation(np.asarray(diffuse), np.asarray(direct))

    assert np.allclose(out.values, ref, equal_nan=True)
    if np.isscalar(diffuse) and np.isscalar(direct):
        assert out.ndim == 0


def test_xr_downward_shortwave_radiation_dims_and_attrs():
    diffuse = _da(DIFFUSE, dims=("point",))
    direct = _da(DIRECT, dims=("point",))

    out = radiation.surface_downward_shortwave_radiation(diffuse, direct)

    assert out.dims == ("point",)
    assert out.attrs["standard_name"] == "surface_downwelling_shortwave_flux_in_air"
    assert out.attrs["units"] == "W m-2"


NET_LONGWAVE = [-60.0, -30.0, -105.5]  # W/m2
SURFACE_TEMPERATURE = [280.0, 300.0, 265.3]  # K


@pytest.mark.parametrize(
    "net_longwave,surface_temperature",
    [
        (NET_LONGWAVE, SURFACE_TEMPERATURE),
        (-60.0, 280.0),
    ],
)
def test_xr_surface_downward_longwave_flux(net_longwave, surface_temperature):
    out = radiation.surface_downward_longwave_flux(_da(net_longwave), _da(surface_temperature))
    ref = radiation.array.surface_downward_longwave_flux(np.asarray(net_longwave), np.asarray(surface_temperature))

    assert np.allclose(out.values, ref, equal_nan=True)
    if np.isscalar(net_longwave) and np.isscalar(surface_temperature):
        assert out.ndim == 0


def test_xr_surface_downward_longwave_flux_emissivity():
    """The emissivity keyword is forwarded to the array backend."""
    net_longwave = _da(NET_LONGWAVE)
    surface_temperature = _da(SURFACE_TEMPERATURE)

    out = radiation.surface_downward_longwave_flux(net_longwave, surface_temperature, emissivity=0.9)
    ref = radiation.array.surface_downward_longwave_flux(
        np.array(NET_LONGWAVE), np.array(SURFACE_TEMPERATURE), emissivity=0.9
    )

    assert np.allclose(out.values, ref)
    assert not np.allclose(out.values, radiation.surface_downward_longwave_flux(net_longwave, surface_temperature))


def test_xr_surface_downward_longwave_flux_dims_and_attrs():
    net_longwave = _da(NET_LONGWAVE, dims=("point",))
    surface_temperature = _da(SURFACE_TEMPERATURE, dims=("point",))

    out = radiation.surface_downward_longwave_flux(net_longwave, surface_temperature)

    assert out.dims == ("point",)
    assert out.attrs["standard_name"] == "surface_downwelling_longwave_flux_in_air"
    assert out.attrs["units"] == "W m-2"


def test_xr_downward_shortwave_radiation_clipped_lazily():
    """Clipping happens inside the ufunc, so dask arrays are not computed eagerly."""
    dask = pytest.importorskip("dask.array")

    diffuse = _da([-5.0, 1.0, 310.0], dims=("point",)).chunk({"point": 2})
    direct = _da([2.0, -4.0, 590.0], dims=("point",)).chunk({"point": 2})

    out = radiation.surface_downward_shortwave_radiation(diffuse, direct)

    assert isinstance(out.data, dask.Array)
    np.testing.assert_allclose(out.compute().values, np.array([0.0, 0.0, 900.0]))
