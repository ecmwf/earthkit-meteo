import datetime

import numpy as np
import pytest
import xarray as xr

from earthkit.meteo.score.deterministic import error
from earthkit.meteo.score.deterministic import mean_error


@pytest.fixture
def sample_temp_ens_forecast_array():
    """Synthetic 2t forecast array with shape (valid_datetime:2, member:2, lat:3, lon:3)"""
    return np.array(
        [
            [
                [
                    [280.0, 285.0, 290.0],
                    [275.0, 270.0, 265.0],
                    [260.0, 255.0, 250.0],
                ],
                [
                    [281.0, 286.0, 291.0],
                    [276.0, 271.0, 266.0],
                    [261.0, 256.0, 251.0],
                ],
            ],
            [
                [
                    [282.0, 287.0, 292.0],
                    [277.0, 272.0, 267.0],
                    [262.0, 257.0, 252.0],
                ],
                [
                    [283.0, 288.0, 293.0],
                    [278.0, 273.0, 268.0],
                    [263.0, 258.0, 253.0],
                ],
            ],
        ]
    )


@pytest.fixture
def sample_temp_observation_array():
    """Synthetic 2t observation array with shape (valid_datetime:2, lat:3, lon:3)"""
    return np.array(
        [
            [
                [280.1, 284.7, 289.2],
                [275.3, 269.9, 264.4],
                [259.8, 254.6, 249.25],
            ],
            [
                [282.6, 286.2, 291.9],
                [277.4, 271.1, 266.75],
                [261.05, 256.33, 251.8],
            ],
        ]
    )


@pytest.fixture
def sample_error_array(sample_temp_ens_forecast_array, sample_temp_observation_array):
    """Synthetic error array for ensemble forecast and observation arrays."""
    return sample_temp_ens_forecast_array - sample_temp_observation_array[:, np.newaxis, :, :]


@pytest.fixture
def ds_forecast_ensemble(sample_temp_ens_forecast_array):
    """Ensemble forecast xarray dataset with dimensions [valid_datetime, number, latitude, longitude]"""
    latitudes = [40.0, 41.0, 42.0]
    longitudes = [10.0, 11.0, 12.0]
    valid_datetimes = [
        datetime.datetime(2024, 1, 1, 0, 0),
        datetime.datetime(2024, 1, 1, 6, 0),
    ]
    ensemble_numbers = [0, 1]

    ds = xr.Dataset(
        {
            "2t": (
                ["valid_datetime", "number", "latitude", "longitude"],
                sample_temp_ens_forecast_array,
            ),
        },
        coords={
            "latitude": latitudes,
            "longitude": longitudes,
            "valid_datetime": valid_datetimes,
            "number": ensemble_numbers,
        },
    )
    return ds


@pytest.fixture
def ds_forecast_deterministic(ds_forecast_ensemble):
    """Deterministic forecast xarray dataset derived from ensemble mean."""
    ds_deterministic = ds_forecast_ensemble.mean(dim="number")
    return ds_deterministic


@pytest.fixture
def ds_observation(sample_temp_observation_array):
    """Observation xarray dataset with dimensions [valid_datetime, latitude, longitude]"""
    latitudes = [40.0, 41.0, 42.0]
    longitudes = [10.0, 11.0, 12.0]
    valid_datetimes = [
        np.datetime64(datetime.datetime(2024, 1, 1, 0, 0)),
        np.datetime64(datetime.datetime(2024, 1, 1, 6, 0)),
    ]

    ds = xr.Dataset(
        {
            "2t": (
                ["valid_datetime", "latitude", "longitude"],
                sample_temp_observation_array,
            ),
        },
        coords={
            "latitude": latitudes,
            "longitude": longitudes,
            "valid_datetime": valid_datetimes,
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
def ds_ensemble_error_expected(sample_error_array):
    """Creates expected result for ensemble error test."""
    return xr.Dataset(
        {
            "2t": (
                ["valid_datetime", "number", "latitude", "longitude"],
                sample_error_array,
            ),
        },
        coords={
            "latitude": [40.0, 41.0, 42.0],
            "longitude": [10.0, 11.0, 12.0],
            # TODO: Or do we expect valid_datetime?
            "valid_datetime": [
                datetime.datetime(2024, 1, 1, 0, 0),
                datetime.datetime(2024, 1, 1, 6, 0),
            ],
            "number": [0, 1],
        },
    )


def test_error_correct(ds_forecast_ensemble, ds_observation, ds_ensemble_error_expected):
    ds_result = error(
        ds_forecast_ensemble,
        ds_observation,
    )
    xr.testing.assert_allclose(ds_result, ds_ensemble_error_expected)


def test_error_with_dataarray(ds_forecast_ensemble, ds_observation, ds_ensemble_error_expected):
    da_result = error(
        ds_forecast_ensemble["2t"],
        ds_observation["2t"],
    )
    xr.testing.assert_allclose(da_result, ds_ensemble_error_expected["2t"])


def test_error_with_aggregation(ds_forecast_ensemble, ds_observation, ds_ensemble_error_expected):
    # agg_method = "mean", no weights
    ds_result = error(
        ds_forecast_ensemble,
        ds_observation,
        agg_method="mean",
        agg_dim=["latitude", "longitude"],
    )
    ds_expected = ds_ensemble_error_expected.mean(dim=["latitude", "longitude"])
    xr.testing.assert_equal(ds_result, ds_expected)


@pytest.mark.skip("TODO")
def test_error_with_weighted_aggregation(ds_forecast_ensemble, ds_observation):
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
    ds_expected = None  # TODO
    xr.testing.assert_equal(ds_result, ds_expected)


@pytest.mark.skip("TODO: Should we support valid_datetime coord propagation?")
def test_error_with_valid_datetime_coord(ds_observation, ds_ensemble_error_expected):
    ds_forecast = xr.Dataset(
        {
            "2t": (
                ["forecast_reference_time", "step", "latitude", "longitude"],
                [
                    [
                        [[280.0, 285.0, 290.0], [275.0, 270.0, 265.0], [260.0, 255.0, 250.0]],
                        [[282.0, 287.0, 292.0], [277.0, 272.0, 267.0], [262.0, 257.0, 252.0]],
                    ]
                ],
            ),
        },
        coords={
            "forecast_reference_time": [datetime.datetime(2024, 1, 1, 0, 0)],
            "step": [datetime.timedelta(hours=0), datetime.timedelta(hours=6)],
            "latitude": [40.0, 41.0, 42.0],
            "longitude": [10.0, 11.0, 12.0],
        },
    )
    ds_forecast = ds_forecast.assign_coords(
        valid_datetime=ds_forecast["forecast_reference_time"] + ds_forecast["step"]
    )

    ds_result = error(ds_forecast, ds_observation)

    assert set(ds_result.dims) == {"forecast_reference_time", "step", "latitude", "longitude"}
    assert "valid_datetime" in ds_result.coords
    xr.testing.assert_allclose(ds_result, ds_ensemble_error_expected)


def test_error_invalid_agg_method(ds_forecast_ensemble, ds_observation):
    with pytest.raises(AssertionError):
        error(
            ds_forecast_ensemble,
            ds_observation,
            agg_method="sum",
            agg_dim=["latitude", "longitude"],
        )


def test_mean_error(ds_forecast_ensemble, ds_observation, ds_ensemble_error_expected):
    da_result = mean_error(
        ds_forecast_ensemble["2t"],
        ds_observation["2t"],
        over="number",
    )
    da_expected = ds_ensemble_error_expected["2t"].mean(dim="number")
    xr.testing.assert_equal(da_result, da_expected)

    ds_result = mean_error(
        ds_forecast_ensemble,
        ds_observation,
        over="number",
    )
    ds_expected = ds_ensemble_error_expected.mean(dim="number")
    xr.testing.assert_equal(ds_result, ds_expected)

    ds_result = mean_error(
        ds_forecast_ensemble,
        ds_observation,
        over=["latitude", "longitude"],
    )
    ds_expected = ds_ensemble_error_expected.mean(dim=["latitude", "longitude"])
    xr.testing.assert_equal(ds_result, ds_expected)

    # TODO: Add test with weights
