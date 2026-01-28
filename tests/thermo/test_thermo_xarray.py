import numpy as np
import pytest

from earthkit.meteo import thermo
from earthkit.meteo.utils.testing import NO_XARRAY

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})
pytestmark = pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")


def _da(values):
    import xarray as xr
    return xr.DataArray(np.asarray(values))


@pytest.mark.parametrize(
    "t_c,t_k_ref",
    [
        ([-273.15, -40.0, 0.0, 20.0, 100.0, np.nan],
         [0.0, 233.15, 273.15, 293.15, 373.15, np.nan]),
    ],
)
def test_celsius_to_kelvin(t_c, t_k_ref):
    t_k = thermo.celsius_to_kelvin(_da(t_c))
    assert np.allclose(t_k.values, np.asarray(t_k_ref), equal_nan=True)


@pytest.mark.parametrize(
    "t_k,t_c_ref",
    [
        ([0.0, 233.15, 273.15, 293.15, 373.15, np.nan],
         [-273.15, -40.0, 0.0, 20.0, 100.0, np.nan]),
    ],
)
def test_kelvin_to_celsius(t_k, t_c_ref):
    t_c = thermo.kelvin_to_celsius(_da(t_k))
    assert np.allclose(t_c.values, np.asarray(t_c_ref), equal_nan=True)


@pytest.mark.parametrize(
    "w,q_ref",
    [
        (
            [0.0, 1.0, 0.5, 0.1, 1e-6, np.nan, -0.5],
            [0.0, 0.5, 1/3, 1/11, 1e-6/(1+1e-6), np.nan, -1.0],
        )
    ],
)
def test_specific_humidity_from_mixing_ratio(w, q_ref):
    q = thermo.specific_humidity_from_mixing_ratio(_da(w))
    assert np.allclose(q.values, np.asarray(q_ref), equal_nan=True)
