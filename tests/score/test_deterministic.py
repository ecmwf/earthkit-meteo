import datetime

import numpy as np
import pytest
import xarray as xr

from earthkit.meteo.score.deterministic import abs_error
from earthkit.meteo.score.deterministic import cosine_similarity
from earthkit.meteo.score.deterministic import error
from earthkit.meteo.score.deterministic import mean_abs_error
from earthkit.meteo.score.deterministic import mean_error
from earthkit.meteo.score.deterministic import mean_squared_error
from earthkit.meteo.score.deterministic import pearson_correlation
from earthkit.meteo.score.deterministic import root_mean_squared_error
from earthkit.meteo.score.deterministic import squared_error
from earthkit.meteo.score.deterministic import standard_deviation_of_error
from earthkit.meteo.utils.testing import NO_SCORES

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


# TODO: Consolidate aggregation tests with and without weights
# TODO: Check that weights dataarray has correct dims matching agg_dim? (check what scores does)
# TODO: Use more interesting fcst/obs data patterns in tests
# TODO: Test NaN handling in both raw error and aggregation
#       Docstrings should clearly document how NaNs are handled
#       (e.g., skip NaNs in aggregation, propagate NaNs in error calc, etc.)
# TODO: Change some tests to not use lat/lon as "over" dims but use valid_datetime.
#       All "low-level" functions should support any dims and we should verify that.
# TODO: Test with non-existent dimension in agg_dim (should raise)
# TODO: Test single variable vs multiple variables in Dataset
# TODO: Should we support valid_datetime coord propagation?
# TODO: Test cupy backed xarray also


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_error(rng):
    fcst_values = np.arange(18.0).reshape(2, 3, 3)
    error_values = rng.uniform(-5, 5, size=(2, 3, 3))
    obs_values = fcst_values - error_values

    fcst = make_dataset(fcst_values)
    obs = make_dataset(obs_values)

    result = error(fcst, obs)

    expected = make_dataset(error_values)
    xr.testing.assert_allclose(result, expected)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
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


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_error_with_aggregation():
    # Error varies by dimension so we can verify correct aggregation axis
    # Rows (lat): [1,1,1], [2,2,2], [3,3,3] -> mean over lat = [2,2,2]
    # Cols (lon): [1,2,3], [1,2,3], [1,2,3] -> mean over lon = [2,2,2]
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

    # Aggregate over lat only -> should preserve lon dimension
    result_lat = error(fcst, obs, agg_method="mean", agg_dim="latitude")
    expected_lat = xr.Dataset(
        {"2t": (["valid_datetime", "longitude"], np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))},
        coords={"valid_datetime": VALID_DATETIMES, "longitude": LONGITUDES},
    )
    xr.testing.assert_equal(result_lat, expected_lat)

    # Aggregate over lon only -> should preserve lat dimension
    result_lon = error(fcst, obs, agg_method="mean", agg_dim="longitude")
    expected_lon = xr.Dataset(
        {
            "2t": (
                ["valid_datetime", "latitude"],
                np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
            )
        },
        coords={"valid_datetime": VALID_DATETIMES, "latitude": LATITUDES},
    )
    xr.testing.assert_equal(result_lon, expected_lon)

    # Aggregate over both -> single value per timestep
    result_both = error(fcst, obs, agg_method="mean", agg_dim=["latitude", "longitude"])
    expected_both = xr.Dataset(
        {"2t": (["valid_datetime"], np.array([2.0, 3.0]))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_equal(result_both, expected_both)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
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


# @pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
# def test_error_weights_without_agg_dim(fcst, obs):
#     weights = xr.DataArray(
#         np.ones((3, 3)),
#         dims=["latitude", "longitude"],
#         coords={"latitude": LATITUDES, "longitude": LONGITUDES},
#     )
#     with pytest.raises((ValueError, TypeError)):
#         error(fcst, obs, agg_weights=weights)


# @pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
# def test_error_agg_dim_without_agg_method(fcst, obs):
#     # TODO This might be silently ignored, or might raise - decide
#     with pytest.raises((ValueError)):
#         error(fcst, obs, agg_dim=["latitude"])


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
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


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
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


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_abs_error():
    fcst_values = np.full((2, 3, 3), 10.0)
    error_values = np.array(
        [
            [[1, -1, 2], [-2, 3, -3], [4, -4, 5]],
            [[-1, 2, -2], [3, -3, 4], [-4, 5, -5]],
        ],
        dtype=float,
    )
    obs_values = fcst_values - error_values

    fcst = make_dataset(fcst_values)
    obs = make_dataset(obs_values)

    result = abs_error(fcst, obs)

    expected = make_dataset(np.abs(error_values))
    xr.testing.assert_equal(result, expected)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_abs_error_is_angular():
    # Wind directions: 350° vs 10° should have error of 20°, not 340°
    fcst_values = np.array(
        [
            [[350, 10, 90], [0, 180, 270], [45, 315, 135]],
            [[0, 90, 180], [270, 45, 135], [315, 225, 0]],
        ],
        dtype=float,
    )
    obs_values = np.array(
        [
            [[10, 350, 270], [180, 0, 90], [315, 45, 315]],
            [[180, 180, 90], [90, 135, 45], [45, 45, 90]],
        ],
        dtype=float,
    )

    fcst = make_dataset(fcst_values)
    obs = make_dataset(obs_values)

    result = abs_error(fcst, obs, is_angular=True)

    # Timestep 1: 20, 20, 180 | 180, 180, 180 | 90, 90, 180
    # Timestep 2: 180, 90, 90 | 180, 90, 90 | 90, 180, 90
    expected_values = np.array(
        [
            [[20, 20, 180], [180, 180, 180], [90, 90, 180]],
            [[180, 90, 90], [180, 90, 90], [90, 180, 90]],
        ],
        dtype=float,
    )
    expected = make_dataset(expected_values)
    xr.testing.assert_equal(result, expected)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_abs_error_with_aggregation():
    fcst_values = np.full((2, 3, 3), 10.0)
    error_values = np.array(
        [
            [[-1, -1, -1], [2, 2, 2], [-3, -3, -3]],
            [[-2, -2, -2], [3, 3, 3], [-4, -4, -4]],
        ],
        dtype=float,
    )
    obs_values = fcst_values - error_values

    fcst = make_dataset(fcst_values)
    obs = make_dataset(obs_values)

    result_lat = abs_error(fcst, obs, agg_method="mean", agg_dim="latitude")
    expected_lat = xr.Dataset(
        {
            "2t": (
                ["valid_datetime", "longitude"],
                np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
            )
        },
        coords={"valid_datetime": VALID_DATETIMES, "longitude": LONGITUDES},
    )
    xr.testing.assert_equal(result_lat, expected_lat)

    result_both = abs_error(fcst, obs, agg_method="mean", agg_dim=["latitude", "longitude"])
    expected_both = xr.Dataset(
        {"2t": (["valid_datetime"], np.array([2.0, 3.0]))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_equal(result_both, expected_both)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_abs_error_with_weighted_aggregation():
    fcst_values = np.full((2, 3, 3), 10.0)
    error_values = np.array(
        [
            [[-1, -1, -1], [-1, -2, -1], [-1, -1, -1]],
            [[-2, -2, -2], [-2, -1, -2], [-2, -2, -2]],
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

    result = abs_error(
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


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_abs_error_invalid_agg_method(fcst, obs):
    with pytest.raises(AssertionError):
        abs_error(fcst, obs, agg_method="sum", agg_dim=["latitude", "longitude"])


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_mean_abs_error():
    fcst_values = np.full((2, 3, 3), 10.0)
    error_values = np.array(
        [
            [[-1, -1, -1], [2, 2, 2], [-3, -3, -3]],
            [[-2, -2, -2], [3, 3, 3], [-4, -4, -4]],
        ],
        dtype=float,
    )
    obs_values = fcst_values - error_values

    fcst = make_dataset(fcst_values)
    obs = make_dataset(obs_values)

    result = mean_abs_error(fcst, obs, over=["latitude", "longitude"])

    # Timestep 1: mean of [1,1,1,2,2,2,3,3,3] = 18/9 = 2.0
    # Timestep 2: mean of [2,2,2,3,3,3,4,4,4] = 27/9 = 3.0
    expected = xr.Dataset(
        {"2t": (["valid_datetime"], np.array([2.0, 3.0]))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_equal(result, expected)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_squared_error():
    fcst_values = np.full((2, 3, 3), 10.0)
    error_values = np.array(
        [
            [[1, -1, 2], [-2, 3, -3], [4, -4, 5]],
            [[-1, 2, -2], [3, -3, 4], [-4, 5, -5]],
        ],
        dtype=float,
    )
    obs_values = fcst_values - error_values

    fcst = make_dataset(fcst_values)
    obs = make_dataset(obs_values)

    result = squared_error(fcst, obs)

    expected = make_dataset(error_values**2)
    xr.testing.assert_equal(result, expected)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_squared_error_is_angular():
    fcst_values = np.array(
        [
            [[350, 10, 90], [0, 180, 270], [45, 315, 135]],
            [[0, 90, 180], [270, 45, 135], [315, 225, 0]],
        ],
        dtype=float,
    )
    obs_values = np.array(
        [
            [[10, 350, 270], [180, 0, 90], [315, 45, 315]],
            [[180, 180, 90], [90, 135, 45], [45, 45, 90]],
        ],
        dtype=float,
    )

    fcst = make_dataset(fcst_values)
    obs = make_dataset(obs_values)

    result = squared_error(fcst, obs, is_angular=True)

    # Timestep 1: 20²=400, 180²=32400, 90²=8100
    # Timestep 2: 180²=32400, 90²=8100
    expected_values = np.array(
        [
            [[400, 400, 32400], [32400, 32400, 32400], [8100, 8100, 32400]],
            [[32400, 8100, 8100], [32400, 8100, 8100], [8100, 32400, 8100]],
        ],
        dtype=float,
    )
    expected = make_dataset(expected_values)
    xr.testing.assert_equal(result, expected)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_squared_error_with_aggregation():
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

    result_both = squared_error(fcst, obs, agg_method="mean", agg_dim=["latitude", "longitude"])
    # Timestep 1: mean of [1,1,1,4,4,4,9,9,9] = 42/9 ≈ 4.667
    # Timestep 2: mean of [4,4,4,9,9,9,16,16,16] = 87/9 ≈ 9.667
    expected_both = xr.Dataset(
        {"2t": (["valid_datetime"], np.array([42 / 9, 87 / 9]))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_allclose(result_both, expected_both)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_squared_error_with_weighted_aggregation():
    fcst_values = np.full((2, 3, 3), 10.0)
    error_values = np.array(
        [
            [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 1, 2], [2, 2, 2]],
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

    result = squared_error(
        fcst,
        obs,
        agg_method="mean",
        agg_dim=["latitude", "longitude"],
        agg_weights=weights,
    )

    # Timestep 1: (8*1 + 1*4*2) / 10 = 16/10 = 1.6
    # Timestep 2: (8*4 + 1*1*2) / 10 = 34/10 = 3.4
    expected = xr.Dataset(
        {"2t": (["valid_datetime"], np.array([1.6, 3.4]))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_allclose(result, expected)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_squared_error_invalid_agg_method(fcst, obs):
    with pytest.raises(AssertionError):
        squared_error(fcst, obs, agg_method="sum", agg_dim=["latitude", "longitude"])


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_mean_squared_error():
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

    result = mean_squared_error(fcst, obs, over=["latitude", "longitude"])

    expected = xr.Dataset(
        {"2t": (["valid_datetime"], np.array([42 / 9, 87 / 9]))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_allclose(result, expected)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_root_mean_squared_error():
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

    result = root_mean_squared_error(fcst, obs, over=["latitude", "longitude"])

    expected = xr.Dataset(
        {"2t": (["valid_datetime"], np.array([np.sqrt(42 / 9), np.sqrt(87 / 9)]))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_allclose(result, expected)


def test_standard_deviation_of_error():
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

    result = standard_deviation_of_error(fcst, obs, over=["latitude", "longitude"])

    # For these patterns both timesteps have variance = 6/9 -> std = sqrt(2/3)
    expected = xr.Dataset(
        {
            "2t": (
                ["valid_datetime"],
                np.array([np.sqrt(2.0 / 3.0), np.sqrt(2.0 / 3.0)]),
            )
        },
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_allclose(result, expected)


def test_standard_deviation_of_error_with_weights():
    fcst_values = np.full((2, 3, 3), 10.0)
    error_values = np.array(
        [
            [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 1, 2], [2, 2, 2]],
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

    result = standard_deviation_of_error(fcst, obs, over=["latitude", "longitude"], weights=weights)

    # Both timesteps have weighted variance 0.16 -> std = 0.4
    expected = xr.Dataset(
        {"2t": (["valid_datetime"], np.array([0.4, 0.4]))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_allclose(result, expected)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_pearson_correlation(rng):
    fcst_values = np.arange(18.0).reshape(2, 3, 3)
    noise = rng.normal(0, 1, size=(2, 3, 3))
    obs_values = fcst_values + noise

    fcst = make_dataset(fcst_values)
    obs = make_dataset(obs_values)

    result = pearson_correlation(fcst, obs, over=["latitude", "longitude"])

    expected_values = []
    for t in range(2):
        fcst_flat = fcst_values[t].flatten()
        obs_flat = obs_values[t].flatten()
        corr = np.corrcoef(fcst_flat, obs_flat)[0, 1]
        expected_values.append(corr)

    expected = xr.Dataset(
        {"2t": (["valid_datetime"], np.array(expected_values))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_allclose(result, expected)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_pearson_correlation_with_weights(rng):
    fcst_values = np.arange(18.0).reshape(2, 3, 3)
    noise = rng.normal(0, 1, size=(2, 3, 3))
    obs_values = fcst_values + noise

    fcst = make_dataset(fcst_values)
    obs = make_dataset(obs_values)

    weights = xr.DataArray(
        np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=float),
        dims=["latitude", "longitude"],
        coords={"latitude": LATITUDES, "longitude": LONGITUDES},
    )

    result = pearson_correlation(
        fcst,
        obs,
        over=["latitude", "longitude"],
        weights=weights,
    )

    # TODO: maybe replace this with just the actual values
    expected_values = []
    for t in range(2):
        fcst_flat = fcst_values[t].flatten()
        obs_flat = obs_values[t].flatten()
        w_flat = weights.values.flatten()
        w_mean_fcst = np.average(fcst_flat, weights=w_flat)
        w_mean_obs = np.average(obs_flat, weights=w_flat)
        cov = np.average((fcst_flat - w_mean_fcst) * (obs_flat - w_mean_obs), weights=w_flat)
        std_fcst = np.sqrt(np.average((fcst_flat - w_mean_fcst) ** 2, weights=w_flat))
        std_obs = np.sqrt(np.average((obs_flat - w_mean_obs) ** 2, weights=w_flat))
        corr = cov / (std_fcst * std_obs)
        expected_values.append(corr)

    expected = xr.Dataset(
        {"2t": (["valid_datetime"], np.array(expected_values))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_allclose(result, expected)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_cosine_similarity():
    fcst = make_dataset(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
        ]
    )
    # Add a bias of 10 and some noise on top
    obs = fcst + make_dataset(
        [
            [
                [10.8784503, 9.95007409, 9.81513764],
                [9.31907046, 11.22254134, 9.84547052],
                [9.57167218, 9.64786645, 10.53230919],
            ],
            [
                [10.36544406, 10.41273261, 10.430821],
                [12.1416476, 9.59358498, 9.48775727],
                [9.18622727, 10.61597942, 11.12897229],
            ],
        ]
    )

    result = cosine_similarity(fcst, obs, over=["latitude", "longitude"])
    expected_values = np.asarray([0.9208026403441584, 0.9954127943028712])
    expected = xr.Dataset(
        {"2t": (["valid_datetime"], np.array(expected_values))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_allclose(result, expected)

    # cosine similarity is commutative
    result = cosine_similarity(obs, fcst, over=["latitude", "longitude"])
    xr.testing.assert_allclose(result, expected)


@pytest.mark.skipif(NO_SCORES, reason="Scores tests disabled")
def test_cosine_similarity_with_weights():
    fcst = make_dataset(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
        ]
    )
    # Add a bias of 10 and some noise on top
    obs = fcst + make_dataset(
        [
            [
                [10.8784503, 9.95007409, 9.81513764],
                [9.31907046, 11.22254134, 9.84547052],
                [9.57167218, 9.64786645, 10.53230919],
            ],
            [
                [10.36544406, 10.41273261, 10.430821],
                [12.1416476, 9.59358498, 9.48775727],
                [9.18622727, 10.61597942, 11.12897229],
            ],
        ]
    )
    weights = xr.DataArray(
        np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=float),
        dims=["latitude", "longitude"],
        coords={"latitude": LATITUDES, "longitude": LONGITUDES},
    )

    result = cosine_similarity(fcst, obs, over=["latitude", "longitude"], weights=weights)
    expected_values = [0.9258378100417775, 0.9958285609895058]
    expected = xr.Dataset(
        {"2t": (["valid_datetime"], np.array(expected_values))},
        coords={"valid_datetime": VALID_DATETIMES},
    )
    xr.testing.assert_allclose(result, expected)
