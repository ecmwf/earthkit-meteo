import datetime

import numpy as np
import pytest
import xarray as xr

from earthkit.meteo.score.metrics import error


@pytest.fixture(autouse=True)
def numpy_random_seed():
    # TODO: Probably this should be done more gracefully
    np.random.seed(42)


@pytest.fixture
def ds_forecast_ensemble():
    """Creates a dummy xarray dataset
    with dimensions [latitude, longitude, number, forecast_issue_time, step]
    and variables '2t', 'tp'.

    grid is 3x3 with latitudes [40, 41, 42] and longitudes [10, 11, 12] (floats).

    Use random normal noise to fill the data arrays.
    """
    latitudes = [40.0, 41.0, 42.0]
    longitudes = [10.0, 11.0, 12.0]
    numbers = list(range(3))
    valid_datetime = [
        datetime.datetime(2023, 1, 1, 0),
        datetime.datetime(2023, 1, 1, 6),
        datetime.datetime(2023, 1, 1, 12),
    ]
    temp = np.random.normal(
        loc=280,
        scale=5,
        size=(len(latitudes), len(longitudes), len(numbers), len(valid_datetime)),
    )
    precip = np.random.normal(
        loc=1,
        scale=0.5,
        size=(len(latitudes), len(longitudes), len(numbers), len(valid_datetime)),
    )
    ds = xr.Dataset(
        {
            "2t": (["latitude", "longitude", "number", "forecast_issue_time"], temp),
            "tp": (["latitude", "longitude", "number", "forecast_issue_time"], precip),
        },
        coords={
            "latitude": latitudes,
            "longitude": longitudes,
            "number": numbers,
            "forecast_issue_time": valid_datetime,
        },
    )
    return ds


@pytest.fixture
def ds_forecast_deterministic(ds_forecast_ensemble):
    # Create a deterministic forecast by selecting number=0
    ds_deterministic = ds_forecast_ensemble.sel(number=0, drop=True)
    return ds_deterministic


@pytest.fixture
def ds_observation(ds_forecast_deterministic):
    """Creates a dummy xarray dataset
    with dimensions [latitude, longitude, forecast_issue_time]
    and variables '2t', 'tp'.

    grid is 3x3 with latitudes [40, 41, 42] and longitudes [10, 11, 12] (floats).

    Use random normal noise to fill the data arrays.
    """
    latitudes = ds_forecast_deterministic.latitude.values
    longitudes = ds_forecast_deterministic.longitude.values
    valid_datetime = ds_forecast_deterministic.forecast_issue_time.values

    temp = np.random.normal(
        loc=280,
        scale=5,
        size=(len(latitudes), len(longitudes), len(valid_datetime)),
    )
    precip = np.random.normal(
        loc=1,
        scale=0.5,
        size=(len(latitudes), len(longitudes), len(valid_datetime)),
    )
    ds = xr.Dataset(
        {
            "2t": (["latitude", "longitude", "forecast_issue_time"], temp),
            "tp": (["latitude", "longitude", "forecast_issue_time"], precip),
        },
        coords={
            "latitude": latitudes,
            "longitude": longitudes,
            "forecast_issue_time": valid_datetime,
        },
    )
    return ds


@pytest.fixture
def da_error_weights():
    """Creates weights for testing weighted aggregation."""
    weights = xr.DataArray(
        np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.3, 0.3, 0.4]]),
        dims=["latitude", "longitude"],
    )
    return weights


@pytest.fixture
def ds_ensemble_error_expected(ds_forecast_ensemble, ds_observation):
    """Creates expected result for ensemble error test."""
    return ds_forecast_ensemble - ds_observation


@pytest.fixture
def ds_ensemble_error_mean_expected(ds_forecast_ensemble, ds_observation):
    """Creates expected result for ensemble error test with mean aggregation."""
    error = ds_forecast_ensemble - ds_observation
    return error.mean(dim=["latitude", "longitude"])


@pytest.fixture
def ds_ensemble_error_mean_weighted_expected(ds_forecast_ensemble, ds_observation):
    """Creates expected result for ensemble error test with mean aggregation and weights."""
    error = ds_forecast_ensemble - ds_observation
    weights = xr.DataArray(
        np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.3, 0.3, 0.4]]),
        dims=["latitude", "longitude"],
    )
    weighted_error = error * weights
    return weighted_error.sum(dim=["latitude", "longitude"]) / weights.sum(dim=["latitude", "longitude"])


# TODO: Ignores nan handling in tests and implementation
def test_error_correct(ds_forecast_ensemble, ds_observation, da_error_weights, ds_ensemble_error_expected):
    # No aggregation
    ds_result = error(
        ds_forecast_ensemble,
        ds_observation,
    )
    xr.testing.assert_equal(ds_result, ds_ensemble_error_expected)

    # agg_method = "mean", no weights
    ds_result = error(
        ds_forecast_ensemble,
        ds_observation,
        agg_method="mean",
        agg_dim=["latitude", "longitude"],
    )

    # agg_method = "mean", with weights
    weights = xr.DataArray(
        np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.3, 0.3, 0.4]]),
        dims=["latitude", "longitude"],
    )
    ds_result = error(
        ds_forecast_ensemble,
        ds_observation,
        agg_method="mean",
        agg_dim=["latitude", "longitude"],
        agg_weights=weights,
    )
    ds_expected = None  # TODO: Hardcoded expected result
    xr.testing.assert_equal(ds_result, ds_expected)


def test_error_invalid_agg_method(ds_forecast_ensemble, ds_observation):
    with pytest.raises(AssertionError):
        error(
            ds_forecast_ensemble,
            ds_observation,
            agg_method="sum",
            agg_dim=["latitude", "longitude"],
        )
