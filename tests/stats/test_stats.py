# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest

from earthkit.meteo import stats


def test_nanaverage():
    data = np.array([[[4, 2], [4, np.nan]], [[4, 4], [np.nan, 2]]])

    # assert nanaverage=nanmean when no weights:
    np.testing.assert_equal(np.nanmean(data, axis=-1), stats.nanaverage(data, axis=-1))

    # Check that using weights returns target result
    weights = np.array([1, 0.25])
    # calculate weighted average
    target_result = np.average(data, axis=-1, weights=weights)
    # replace nan values
    target_result[:, 1] = np.nansum(data, axis=-1)[:, 1]
    print(target_result)
    np.testing.assert_equal(stats.nanaverage(data, axis=-1, weights=weights), target_result)


@pytest.mark.parametrize("method", ["sort", "numpy_bulk", "numpy"])
def test_quantiles(method):
    data = np.array(
        [
            [3, 6, 12, 0, 45, 0, 4, 0, 7],
            [1, 2.5, 4, 3, 18, 3, 8, 7, 2],
            [3, 4, 9, 0.5, 2, 48, 4, 9, 6],
            [5, 19, 8.3, 2, 6, 1, 0, 1, 0],
        ],
        dtype=np.float64,
    )

    quartiles = list(stats.iter_quantiles(data.copy(), 4, axis=1, method=method))
    assert len(quartiles) == 5
    expectedq = np.array(
        [
            [0, 1, 0.5, 0],
            [0, 2.5, 3, 1],
            [4, 3, 4, 2],
            [7, 7, 9, 6],
            [45, 18, 48, 19],
        ],
        dtype=np.float64,
    )
    np.testing.assert_equal(quartiles, expectedq)

    quantiles = list(stats.iter_quantiles(data.copy(), [0.5, 1.0], method=method))
    assert len(quantiles) == 2
    expectedq = np.array(
        [
            [3, 5, 6.15, 1.25, 12, 2, 4, 4, 4],
            [5, 19, 12, 3, 45, 48, 8, 9, 7],
        ]
    )


def test_quantiles_nans():
    arr = np.random.rand(100, 100, 100)
    arr.ravel()[np.random.choice(arr.size, 100000, replace=False)] = np.nan
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    sort = [quantile for quantile in stats.iter_quantiles(arr.copy(), qs, method="sort")]
    numpy = [quantile for quantile in stats.iter_quantiles(arr.copy(), qs, method="numpy")]
    assert np.all(np.isclose(sort, numpy, equal_nan=True))


def test_GumbelDistribution():
    dist = stats.GumbelDistribution.fit([6.0, 5.0, 6.0, 7.0, 9.0, 5.0, 6.0, 7.0])
    # Extreme ends of distribution
    np.testing.assert_allclose(dist.cdf(0.0), 1.0)
    np.testing.assert_allclose(dist.cdf(100.0), 0.0)
    # Test identity when calling method followed by inverse method
    values = np.linspace(4.0, 10.0, 21)
    np.testing.assert_allclose(dist.ppf(dist.cdf(values)), values)
    probs = np.linspace(0.01, 0.99, 21)
    np.testing.assert_allclose(dist.cdf(dist.ppf(probs)), probs)


def test_GumbelDistribution_along_axis():
    sample = [
        [0.3, 3.0, 30.0],
        [0.5, 5.0, 50.0],
        [0.7, 7.0, 70.0],
        [0.3, 3.0, 30.0],
        [0.4, 4.0, 40.0],
        [0.6, 6.0, 60.0],
        [0.7, 7.0, 70.0],
    ]
    dist = stats.GumbelDistribution.fit(sample, axis=0)
    values = dist.ppf([0.01, 0.3, 0.5, 0.6, 0.99])
    assert values.shape == (5, 3)
    # Results should scale with values
    np.testing.assert_allclose(values[:, 0] * 10.0, values[:, 1])
    np.testing.assert_allclose(values[:, 1] * 10.0, values[:, 2])


def test_return_period_identity():
    dist = stats.GumbelDistribution.fit([6.0, 5.0, 6.0, 7.0, 9.0, 5.0, 6.0, 7.0])
    values = np.linspace(4.0, 10.0, 21)
    np.testing.assert_allclose(
        stats.return_period_to_value(dist, stats.value_to_return_period(dist, values)), values
    )


def test_return_period_with_frequency():
    freq = np.timedelta64(24 * 60 * 60, "s")
    sample = [4.0, 3.0, 5.5, 6.0, 7.0, 5.3, 2.1]
    dist = stats.GumbelDistribution.fit(sample, freq=freq)
    # Return periods should be scaled by the given frequency
    assert stats.value_to_return_period(dist, 6.0).dtype == freq.dtype
