import xarray as xr
import pytest

from earthkit.meteo import thermo
from earthkit.meteo.utils import testing


@pytest.fixture
def input_ds():
    path = testing.get_test_data("sleve_to_theta_input.nc", "test-data")
    return xr.open_dataset(path)


@pytest.fixture
def output_ds():
    path = testing.get_test_data("sleve_to_theta_ref.nc", "test-data")
    return xr.open_dataset(path)


def test_interpolate_sleve_to_theta_levels(input_ds, output_ds):
    from earthkit.meteo.vertical.interpolation import interpolate_sleve_to_theta_levels

    ds = input_ds.rename({"level": "z"})
    p = ds["P"]
    t = ds["T"]
    h = ds["HFL"]
    target_theta = [280.0, 290.0, 310.0, 315.0, 320.0, 325.0, 330.0, 335.0]

    theta = xr.apply_ufunc(thermo.potential_temperature, t, p)
    observed = interpolate_sleve_to_theta_levels(t, h, theta, target_theta, "K", "high_fold")

    assert observed == output_ds
