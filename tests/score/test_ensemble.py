import datetime

import numpy as np
import pytest
import xarray as xr

from earthkit.meteo.score.ensemble import crps_from_ensemble
from earthkit.meteo.score.ensemble import crps_gaussian
from earthkit.meteo.score.ensemble import quantile_score
from earthkit.meteo.score.ensemble import spread

LATITUDES = [40.0, 41.0]
LONGITUDES = [10.0, 11.0]
VALID_DATETIMES = [
    datetime.datetime(2024, 1, 1, 0, 0),
    datetime.datetime(2024, 1, 1, 6, 0),
]


def make_ensemble_dataset(values, var_name="2t"):
    """Build an ensemble (time, lat, lon, number) dataset."""
    assert len(values.shape) == 4
    numbers = values.shape[3]
    return xr.Dataset(
        {var_name: (["valid_datetime", "latitude", "longitude", "number"], values)},
        coords={
            "valid_datetime": VALID_DATETIMES,
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
            "number": np.arange(numbers),
        },
    )


def make_gaussian_ensemble_dataset(mean, stdev):
    assert len(mean.shape) == 3
    assert len(stdev.shape) == 3
    return xr.Dataset(
        {
            "mean": (["valid_datetime", "latitude", "longitude"], mean),
            "stdev": (["valid_datetime", "latitude", "longitude"], stdev),
        },
        coords={
            "valid_datetime": VALID_DATETIMES,
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
        },
    )


def make_deterministic_dataset(values, var_name="2t"):
    """Build a deterministic (time, lat, lon) dataset."""
    return xr.Dataset(
        {var_name: (["valid_datetime", "latitude", "longitude"], values)},
        coords={
            "valid_datetime": VALID_DATETIMES,
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
        },
    )


def make_deterministic_dataarray(values, var_name="2t"):
    """Build a deterministic (time, lat, lon) dataarray."""
    return xr.DataArray(
        values,
        dims=["valid_datetime", "latitude", "longitude"],
        coords={
            "valid_datetime": VALID_DATETIMES,
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
        },
        name=var_name,
    )


def assert_component_allclose(components: xr.DataArray, component: str, expected_values: np.ndarray):
    expected = make_deterministic_dataarray(expected_values, var_name=component)
    computed = components.sel(component=component, drop=True).rename(component)
    xr.testing.assert_allclose(computed, expected)


@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


def test_spread_without_reference():
    """Test spread calculation without a reference, using the ensemble mean."""

    # Forecast data: shape (time, lat, lon, number)
    fcst_values = np.array(
        [
            [
                [[10, 12, 14], [20, 22, 24]],
                [[30, 32, 34], [40, 42, 44]],
            ],
            [
                [[11, 13, 15], [21, 23, 25]],
                [[31, 33, 35], [41, 43, 45]],
            ],
        ]
    )
    spread_values = np.std(fcst_values, axis=3, ddof=0)

    fcst = make_ensemble_dataset(fcst_values)
    spread_expected = make_deterministic_dataset(spread_values)

    spread_computed = spread(fcst, over="number")
    xr.testing.assert_allclose(spread_computed, spread_expected)


def test_spread_with_reference():
    fcst_values = np.array(
        [
            [
                [[10, 12, 14], [20, 22, 24]],
                [[30, 32, 34], [40, 42, 44]],
            ],
            [
                [[11, 13, 15], [21, 23, 25]],
                [[31, 33, 35], [41, 43, 45]],
            ],
        ]
    )
    reference_values = np.array(
        [
            [[12, 24], [36, 48]],
            [[7, 19], [31, 43]],
        ]
    )
    spread_values = np.sqrt(
        np.mean(
            (fcst_values - reference_values[:, :, :, np.newaxis]) ** 2,
            axis=3,
        )
    )
    fcst = make_ensemble_dataset(fcst_values)
    reference = make_deterministic_dataset(reference_values)
    spread_expected = make_deterministic_dataset(spread_values)
    spread_computed = spread(fcst, over="number", reference=reference)
    xr.testing.assert_allclose(spread_computed, spread_expected)

    # DataArrays
    spread_computed_da = spread(fcst["2t"], over="number", reference=reference["2t"])
    xr.testing.assert_allclose(spread_computed_da, spread_expected["2t"])

    # Mixed
    spread_computed_mixed = spread(fcst, over="number", reference=reference["2t"])
    xr.testing.assert_allclose(spread_computed_mixed, spread_expected)

    # Reference with 'number' dimension squeezed
    reference_squeezed = reference.expand_dims({"number": [0]})
    spread_computed_squeezed = spread(fcst, over="number", reference=reference_squeezed)
    xr.testing.assert_allclose(spread_computed_squeezed, spread_expected)


@pytest.mark.parametrize(
    "tau,expected_scores",
    [
        (0.1, np.array([[[1.98, 0.18], [0.18, 0.38]], [[0.08, 0.0], [41.58, 0.38]]])),
        (0.5, np.array([[[1.5, 0.5], [0.5, 1.5]], [[0.0, 0.4], [23.5, 1.5]]])),
        (0.9, np.array([[[0.38, 0.18], [0.18, 1.98]], [[0.08, 0.16], [4.78, 1.98]]])),
    ],
)
def test_quantile_score(tau: float, expected_scores: np.ndarray):
    # Forecast data: shape (time, lat, lon, number)
    fcst_values = np.array(
        [
            [
                [np.linspace(0.0, 1.0, 10), np.linspace(10.0, 11.0, 10)],
                [np.linspace(20.0, 21.0, 10), np.linspace(30.0, 31.0, 10)],
            ],
            [
                [np.linspace(1.0, 2.0, 10), np.linspace(11.0, 12.0, 10)],
                [np.linspace(21.0, 22.0, 10), np.linspace(31.0, 32.0, 10)],
            ],
        ]
    )
    obs_values = np.array(
        [
            [
                [-1.0, 10.0],
                [21.0, 32.0],
            ],
            [
                [1.5, 11.1],
                [-2.0, 33.0],
            ],
        ]
    )

    fcst = make_ensemble_dataset(fcst_values)
    obs = make_deterministic_dataset(obs_values)
    qs_expected = make_deterministic_dataset(expected_scores)

    # Datasets
    qs_computed = quantile_score(fcst, obs, tau=tau, over="number")
    xr.testing.assert_allclose(qs_computed, qs_expected)

    # DataArrays
    qs_computed_da = quantile_score(fcst["2t"], obs["2t"], tau=tau, over="number")
    xr.testing.assert_allclose(qs_computed_da, qs_expected["2t"])


@pytest.mark.parametrize(
    "tau",
    [-0.1, 0.0, 1.0, 1.1],
)
def test_quantile_score_invalid_tau(tau: float):
    fcst_values = np.random.rand(2, 2, 2, 10)
    obs_values = np.random.rand(2, 2, 2)

    fcst = make_ensemble_dataset(fcst_values)
    obs = make_deterministic_dataset(obs_values)

    with pytest.raises(ValueError, match="tau must be in the range"):
        quantile_score(fcst, obs, tau=tau, over="number")


def test_crps_gaussian():
    fcst_mean_values = np.array(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[15.0, 25.0], [35.0, 45.0]],
        ]
    )
    fcst_stddev_values = np.array(
        [
            [[2.0, 3.0], [4.0, 5.0]],
            [[2.5, 3.5], [4.5, 5.5]],
        ]
    )
    obs_values = np.array(
        [
            [[12.0, 19.0], [29.0, 42.0]],
            [[14.0, 27.0], [36.0, 44.0]],
        ]
    )
    expected_crps_values = np.array(
        [
            [[1.20488272, 0.83284794], [1.03399925, 1.48344045]],
            [[0.74172023, 1.26185368], [1.1399182, 1.35765817]],
        ]
    )

    # TODO: Settle on stdev vs stddev
    fcst = make_gaussian_ensemble_dataset(fcst_mean_values, fcst_stddev_values)
    obs = make_deterministic_dataset(obs_values)["2t"]
    crps_expected = make_deterministic_dataset(expected_crps_values)["2t"]

    crps_computed = crps_gaussian(fcst, obs)
    xr.testing.assert_allclose(crps_computed, crps_expected)


def test_crps_gaussian_invalid_input():
    fcst_mean_values = np.array(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[15.0, 25.0], [35.0, 45.0]],
        ]
    )
    # Missing 'stdev' variable
    fcst = xr.Dataset(
        {
            "mean": (["valid_datetime", "latitude", "longitude"], fcst_mean_values),
        },
        coords={
            "valid_datetime": VALID_DATETIMES,
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
        },
    )
    obs_values = np.array(
        [
            [[12.0, 19.0], [29.0, 42.0]],
            [[14.0, 27.0], [36.0, 44.0]],
        ]
    )
    obs = make_deterministic_dataset(obs_values)["2t"]

    with pytest.raises(ValueError, match="Expected fcst to have 'mean' and 'stdev' data variables"):
        crps_gaussian(fcst, obs)

    with pytest.raises(TypeError, match="Expected fcst to be an xarray.Dataset object"):
        crps_gaussian(
            make_gaussian_ensemble_dataset(fcst_mean_values, fcst_mean_values)["mean"],
            make_deterministic_dataset(obs_values),
        )

    with pytest.raises(TypeError, match="Expected obs to be an xarray.DataArray object"):
        crps_gaussian(
            make_gaussian_ensemble_dataset(fcst_mean_values, fcst_mean_values),
            make_deterministic_dataset(obs_values),
        )


def test_crps_from_ensemble_invalid_method():
    fcst_values = np.random.rand(2, 2, 2, 10)
    obs_values = np.random.rand(2, 2, 2)

    fcst = make_ensemble_dataset(fcst_values)
    obs = make_deterministic_dataset(obs_values)["2t"]

    with pytest.raises(ValueError, match="must be one of"):
        crps_from_ensemble(fcst, obs, over="number", method="invalid_method")


@pytest.mark.parametrize(
    "method,expected_total,expected_spread",
    [
        (
            "ecdf",
            np.array(
                [
                    [[2.0, 0.2], [2.4, 6.2]],
                    [[2.8, 0.46], [0.4, 0.0]],
                ],
                dtype=float,
            ),
            np.array(
                [
                    [[0.0, 0.4], [4.0, 0.8]],
                    [[0.4, 0.64], [0.8, 0.0]],
                ],
                dtype=float,
            ),
        ),
        (
            "fair",
            np.array(
                [
                    [[2.0, 0.1], [1.4, 6.0]],
                    [[2.7, 0.3], [0.2, 0.0]],
                ],
                dtype=float,
            ),
            np.array(
                [
                    [[0.0, 0.5], [5.0, 1.0]],
                    [[0.5, 0.8], [1.0, 0.0]],
                ],
                dtype=float,
            ),
        ),
    ],
)
def test_crps_from_ensemble(method: str, expected_total: np.ndarray, expected_spread: np.ndarray):
    fcst_values = np.array(
        [
            [
                [
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [1.0, 2.0, 0.0, 2.0, 1.0],
                ],
                [
                    [-5.0, 0.0, 5.0, 10.0, 15.0],
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                ],
            ],
            [
                [
                    [-3.0, -2.0, -1.0, -1.0, -2.0],
                    [4.0, 1.0, 3.0, 1.0, 2.0],
                ],
                [
                    [-2.0, -1.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
        ],
        dtype=float,
    )
    obs_values = np.array(
        [
            [
                [0.0, 1.0],
                [7.0, 10.0],
            ],
            [
                [-5.0, 2.5],
                [0.0, 0.0],
            ],
        ],
        dtype=float,
    )

    expected_under = np.array(
        [
            [[0.0, 0.2], [4.2, 7.0]],
            [[0.0, 0.7], [0.6, 0.0]],
        ],
        dtype=float,
    )
    expected_over = np.array(
        [
            [[2.0, 0.4], [2.2, 0.0]],
            [[3.2, 0.4], [0.6, 0.0]],
        ],
        dtype=float,
    )

    fcst_ds = make_ensemble_dataset(fcst_values)
    obs_ds = make_deterministic_dataset(obs_values)

    crps_expected_ds = make_deterministic_dataset(expected_total)
    crps_computed_ds = crps_from_ensemble(fcst_ds, obs_ds, over="number", method=method)
    xr.testing.assert_allclose(crps_computed_ds, crps_expected_ds)

    fcst_da = fcst_ds["2t"]
    obs_da = obs_ds["2t"]
    crps_expected_da = crps_expected_ds["2t"]

    crps_computed_da = crps_from_ensemble(fcst_da, obs_da, over="number", method=method)
    xr.testing.assert_allclose(crps_computed_da, crps_expected_da)

    expected_components = {
        "total": expected_total,
        "underforecast_penalty": expected_under,
        "overforecast_penalty": expected_over,
        "spread": expected_spread,
    }

    crps_components = crps_from_ensemble(
        fcst_da,
        obs_da,
        over="number",
        method=method,
        return_components=True,
    )

    assert list(crps_components["component"].values) == [
        "total",
        "underforecast_penalty",
        "overforecast_penalty",
        "spread",
    ]

    for component, values in expected_components.items():
        assert_component_allclose(crps_components, component, values)

    total = crps_components.sel(component="total", drop=True)
    under = crps_components.sel(component="underforecast_penalty", drop=True)
    over = crps_components.sel(component="overforecast_penalty", drop=True)
    spread = crps_components.sel(component="spread", drop=True)
    xr.testing.assert_allclose(total, under + over - spread)
