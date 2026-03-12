# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import numpy as np
import pytest

from earthkit.meteo.stats.array import fit_gumbel
from earthkit.meteo.stats.array import return_period_to_value
from earthkit.meteo.stats.array import value_to_return_period


@pytest.fixture
def gumbel_0d():
    return fit_gumbel([6.0, 5.0, 6.0, 7.0, 9.0, 5.0, 6.0, 7.0])


@pytest.fixture
def gumbel_1d():
    return fit_gumbel(
        [
            [0.3, 3.0, 30.0],
            [0.5, 5.0, 50.0],
            [0.7, 7.0, 70.0],
            [0.3, 3.0, 30.0],
            [0.4, 4.0, 40.0],
            [0.6, 6.0, 60.0],
            [0.7, 7.0, 70.0],
        ],
        dim=0,
    )


def test_GumbelDistribution_from_fit_gumbel_0d(gumbel_0d):
    assert gumbel_0d.ndim == 0
    assert gumbel_0d.shape == tuple()
    # Extreme ends of distribution
    np.testing.assert_allclose(gumbel_0d.cdf(-np.inf), 1.0)
    np.testing.assert_allclose(gumbel_0d.cdf(np.inf), 0.0)
    # Test identity when calling method followed by inverse method
    values = np.linspace(4.0, 10.0, 21)
    np.testing.assert_allclose(gumbel_0d.ppf(gumbel_0d.cdf(values)), values)
    probs = np.linspace(0.01, 0.99, 21)
    np.testing.assert_allclose(gumbel_0d.cdf(gumbel_0d.ppf(probs)), probs)


def test_GumbelDistribution_from_fit_gumbel_1d(gumbel_1d):
    assert gumbel_1d.ndim == 1
    assert gumbel_1d.shape == (3,)
    values = gumbel_1d.ppf([0.01, 0.3, 0.5, 0.6, 0.99])
    assert values.shape == (5, 3)
    # Results should scale with values
    np.testing.assert_allclose(values[:, 0] * 10.0, values[:, 1])
    np.testing.assert_allclose(values[:, 1] * 10.0, values[:, 2])


def test_return_period_identity(gumbel_0d):
    values = np.linspace(4.0, 10.0, 21)
    np.testing.assert_allclose(
        values, return_period_to_value(value_to_return_period(values, gumbel_0d), gumbel_0d)
    )


def test_value_to_return_period_adds_dist_dims_at_end(gumbel_0d, gumbel_1d):
    values = 2.0 * np.ones((2, 1))
    assert value_to_return_period(values, gumbel_0d).shape == (2, 1)
    assert value_to_return_period(values, gumbel_1d).shape == (2, 1, 3)


def test_return_period_to_value_adds_dist_dims_at_end(gumbel_0d, gumbel_1d):
    rps = 3.0 * np.ones((2, 1))
    assert return_period_to_value(rps, gumbel_0d).shape == (2, 1)
    assert return_period_to_value(rps, gumbel_1d).shape == (2, 1, 3)
