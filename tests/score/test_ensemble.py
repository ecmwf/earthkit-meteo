import datetime

import numpy as np
import pytest
import xarray as xr

from earthkit.meteo.score.ensemble import spread

LATITUDES = [40.0, 41.0]
LONGITUDES = [10.0, 11.0]
NUMBERS = [0, 1, 2]
VALID_DATETIMES = [
    datetime.datetime(2024, 1, 1, 0, 0),
    datetime.datetime(2024, 1, 1, 6, 0),
]


def make_ensemble_dataset(values, var_name="2t"):
    """Build an ensemble (time, lat, lon, number) dataset."""
    return xr.Dataset(
        {var_name: (["valid_datetime", "latitude", "longitude", "number"], values)},
        coords={
            "valid_datetime": VALID_DATETIMES,
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
            "number": NUMBERS,
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

    # Datasets
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
