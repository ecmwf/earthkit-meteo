import datetime

import numpy as np
import pytest
import xarray as xr

from earthkit.meteo.score.deterministic import error
from earthkit.meteo.score.deterministic import mean_error

LATITUDES = [40.0, 41.0, 42.0]
LONGITUDES = [10.0, 11.0, 12.0]
VALID_DATETIMES = [
    datetime.datetime(2024, 1, 1, 0, 0),
    datetime.datetime(2024, 1, 1, 6, 0),
]


def make_dataset(values, var_name="2t"):
    """Build a standard (time, lat, lon) dataset."""
    return xr.Dataset(
        {var_name: (["valid_datetime", "latitude", "longitude"], values)},
        coords={
            "valid_datetime": VALID_DATETIMES,
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
        },
    )


@pytest.fixture
def fcst():
    """Forecast: sequential values 0-17."""
    return make_dataset(np.arange(18.0).reshape(2, 3, 3))


@pytest.fixture
def obs():
    """Observation: sequential values 1-18, so error = fcst - obs = -1 everywhere."""
    return make_dataset(np.arange(1.0, 19.0).reshape(2, 3, 3))


@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


def test_error(rng):
    fcst_values = np.arange(18.0).reshape(2, 3, 3)
    error_values = rng.uniform(-5, 5, size=(2, 3, 3))
    obs_values = fcst_values - error_values

    fcst = make_dataset(fcst_values)
    obs = make_dataset(obs_values)

    result = error(fcst, obs)

    expected = make_dataset(error_values)
    xr.testing.assert_allclose(result, expected)


def test_error_accepts_dataarray(rng):
    """Verify API accepts both Dataset and DataArray input."""
    fcst_values = np.arange(18.0).reshape(2, 3, 3)
    error_values = rng.uniform(-5, 5, size=(2, 3, 3))
    obs_values = fcst_values - error_values

    fcst_ds = make_dataset(fcst_values)
    obs_ds = make_dataset(obs_values)

    result_da = error(fcst_ds["2t"], obs_ds["2t"])
    assert isinstance(result_da, xr.DataArray)
    assert result_da.shape == (2, 3, 3)


def test_error_with_aggregation():
    # Error varies by dimension so we can verify correct aggregation axis
    # Rows (lat): [1,1,1], [2,2,2], [3,3,3] -> mean over lat = [2,2,2]
    # Cols (lon): [1,2,3], [1,2,3], [1,2,3] -> mean over lon = [2,2,2]
    fcst_values = np.full((2, 3, 3), 10.0)
    error_values = np.array(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
        ],
        dtype=float,
    )
    obs_values = fcst_values - error_values

    fcst = make_dataset(fcst_values)
    obs = make_dataset(obs_values)

    # Aggregate over lat only -> should preserve lon dimension
    result_lat = error(fcst, obs, agg_method="mean", agg_dim="latitude")
    expected_lat = xr.Dataset(
        {"2t": (["valid_datetime", "longitude"], np.full((2, 3), 2.0))},
        coords={"valid_datetime": VALID_DATETIMES, "longitude": LONGITUDES},
    )
    xr.testing.assert_equal(result_lat, expected_lat)

    # Aggregate over lon only -> should preserve lat dimension
    result_lon = error(fcst, obs, agg_method="mean", agg_dim="longitude")
    expected_lon = xr.Dataset(
        {"2t": (["valid_datetime", "latitude"], np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))},
        coords={"valid_datetime": VALID_DATETIMES, "latitude": LATITUDES},
    )
    xr.testing.assert_equal(result_lon, expected_lon)

    # Aggregate over both -> single value per timestep
    result_both = error(fcst, obs, agg_method="mean", agg_dim=["latitude", "longitude"])
    expected_both = xr.Dataset(
        {"2t": (["valid_datetime"], np.array([2.0, 2.0]))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_equal(result_both, expected_both)


def test_error_with_weighted_aggregation():
    fcst_values = np.full((2, 3, 3), 10.0)
    error_values = np.array(
        [
            [[1, 1, 1], [1, 2, 1], [1, 1, 1]],  # First timestep: mostly 1, center is 2
            [[2, 2, 2], [2, 1, 2], [2, 2, 2]],  # Second timestep: mostly 2, center is 1
        ],
        dtype=float,
    )
    obs_values = fcst_values - error_values

    fcst = make_dataset(fcst_values)
    obs = make_dataset(obs_values)

    weights = xr.DataArray(
        np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=float),
        dims=["latitude", "longitude"],
        coords={"latitude": LATITUDES, "longitude": LONGITUDES},
    )

    result = error(
        fcst,
        obs,
        agg_method="mean",
        agg_dim=["latitude", "longitude"],
        agg_weights=weights,
    )

    # Timestep 1: (8*1 + 1*2*2) / 10 = 12/10 = 1.2
    # Timestep 2: (8*2 + 1*1*2) / 10 = 18/10 = 1.8
    expected = xr.Dataset(
        {"2t": (["valid_datetime"], np.array([1.2, 1.8]))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_allclose(result, expected)


def test_error_invalid_agg_method(fcst, obs):
    with pytest.raises(AssertionError):
        error(fcst, obs, agg_method="sum", agg_dim=["latitude", "longitude"])


@pytest.mark.skip("TODO")
def test_error_weights_without_agg_dim(fcst, obs):
    weights = xr.DataArray(
        np.ones((3, 3)),
        dims=["latitude", "longitude"],
        coords={"latitude": LATITUDES, "longitude": LONGITUDES},
    )
    with pytest.raises((ValueError, TypeError)):
        error(fcst, obs, agg_weights=weights)


@pytest.mark.skip("TODO")
def test_error_agg_dim_without_agg_method(fcst, obs):
    # TODO This might be silently ignored, or might raise - decide
    with pytest.raises((ValueError)):
        error(fcst, obs, agg_dim=["latitude"])


def test_mean_error(rng):
    # Error pattern: rows vary [1,2,3], so mean over lat/lon shows per-timestep variation
    fcst_values = np.full((2, 3, 3), 10.0)
    error_values = np.array(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[2, 2, 2], [3, 3, 3], [4, 4, 4]],
        ],
        dtype=float,
    )
    obs_values = fcst_values - error_values

    fcst = make_dataset(fcst_values)
    obs = make_dataset(obs_values)

    result = mean_error(fcst, obs, over=["latitude", "longitude"])

    # First timestep: mean of [1,1,1,2,2,2,3,3,3] = 18/9 = 2.0
    # Second timestep: mean of [2,2,2,3,3,3,4,4,4] = 27/9 = 3.0
    expected = xr.Dataset(
        {"2t": (["valid_datetime"], np.array([2.0, 3.0]))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_equal(result, expected)


def test_mean_error_with_weights(rng):
    # Error pattern: rows vary [1,2,3], so mean over lat/lon shows per-timestep variation
    fcst_values = np.full((2, 3, 3), 10.0)
    error_values = np.array(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[2, 2, 2], [3, 3, 3], [4, 4, 4]],
        ],
        dtype=float,
    )
    obs_values = fcst_values - error_values

    fcst = make_dataset(fcst_values)
    obs = make_dataset(obs_values)

    weights = xr.DataArray(
        np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=float),
        dims=["latitude", "longitude"],
        coords={"latitude": LATITUDES, "longitude": LONGITUDES},
    )

    result = mean_error(
        fcst,
        obs,
        over=["latitude", "longitude"],
        weights=weights,
    )

    # Timestep 1: (1*1 + 1*1 + 1*1 + 1*2 + 2*2 + 1*2 + 1*3 + 1*3 + 1*3) / 10 = 20/10 = 2.0
    # Timestep 2: (1*2 + 1*2 + 1*2 + 1*3 + 2*3 + 1*3 + 1*4 + 1*4 + 1*4) / 10 = 30/10 = 3.0
    expected = xr.Dataset(
        {"2t": (["valid_datetime"], np.array([2.0, 3.0]))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_allclose(result, expected)


# TODO: Check that weights dataarray has correct dims matching agg_dim?
# TODO: Use more interesting fcst/obs data patterns in tests
# TODO: Test NaN handling in both raw error and aggregation
# TODO: Test with non-existent dimension in agg_dim (should raise)
# TODO: Test single variable vs multiple variables in Dataset
# TODO: Should we support valid_datetime coord propagation?
