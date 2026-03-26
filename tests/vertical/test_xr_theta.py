import numpy as np
import pytest

from earthkit.meteo.thermo.array import potential_temperature
from earthkit.meteo.utils import testing

xr = pytest.importorskip("xarray")


@pytest.fixture
def input_ds():
    path = testing.get_test_data("sleve_to_theta_input.nc", "test-data")
    return xr.open_dataset(path)


@pytest.fixture
def output_ds():
    def f(coord, mode):
        path = testing.get_test_data(f"sleve_to_{coord}_ref_{mode}.nc", "test-data")
        return xr.open_dataset(path)

    return f


@pytest.mark.parametrize("mode", ["high_fold", "low_fold"])
def test_interpolate_sleve_to_theta_levels(mode, input_ds, output_ds):
    from earthkit.meteo.vertical.interpolation import interpolate_sleve_to_theta_levels

    ds = input_ds.rename({"level": "z"})
    p = ds["P"]
    t = ds["T"]
    h = ds["HFL"]
    target_theta = [280.0, 290.0, 310.0, 315.0, 320.0, 325.0, 330.0, 335.0]
    theta = xr.apply_ufunc(potential_temperature, t, p)

    observed = interpolate_sleve_to_theta_levels(t, h, theta, target_theta, "K", folding_mode=mode).values
    expected = output_ds("theta", mode)["T"].values

    np.testing.assert_allclose(observed, expected, 1e-5, 1e-7)


@pytest.mark.parametrize("interpolation", ["linear", "log", "nearest"])
def test_interpolate_to_pressure(interpolation, input_ds, output_ds):
    from earthkit.meteo.vertical.interpolation import interpolate_to_pressure_levels

    ds = input_ds.rename({"level": "z"})
    p = ds["P"]
    t = ds["T"]
    target_p = [40.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1100.0]
    target_p_units = "hPa"

    observed = interpolate_to_pressure_levels(
        t, p, target_p, target_p_units, interpolation=interpolation
    ).values
    expected = output_ds("pressure", interpolation)["T"].values.squeeze()

    np.testing.assert_allclose(observed, expected, 1e-5, 1e-7)
