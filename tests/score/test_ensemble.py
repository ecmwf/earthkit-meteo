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
    """Build a standard (time, lat, lon, number) dataset."""
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
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


def test_spread_without_reference():
    """Test spread calculation without a reference, using the ensemble mean."""

    # Forecast data: shape (time, lat, lon, number)
    fcst_values = np.array(
        [
            [
                [
                    [10, 12, 14],
                    [20, 22, 24],
                ],
                [
                    [30, 32, 34],
                    [40, 42, 44],
                ],
            ],
            [
                [
                    [11, 13, 15],
                    [21, 23, 25],
                ],
                [
                    [31, 33, 35],
                    [41, 43, 45],
                ],
            ],
        ]
    )
    spread_values = np.std(fcst_values, axis=3, ddof=0)

    fcst = make_ensemble_dataset(fcst_values)
    spread_expected = make_deterministic_dataset(spread_values, var_name="2t")

    spread_computed = spread(fcst, over="number")
    xr.testing.assert_allclose(spread_computed, spread_expected)
