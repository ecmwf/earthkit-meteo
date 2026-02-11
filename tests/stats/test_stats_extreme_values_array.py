# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np

from earthkit.meteo.stats import return_period_to_value
from earthkit.meteo.stats import value_to_return_period
from earthkit.meteo.stats.array import GumbelDistribution


def test_GumbelDistributionArray():
    dist = GumbelDistribution.fit([6.0, 5.0, 6.0, 7.0, 9.0, 5.0, 6.0, 7.0])
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
    dist = GumbelDistribution.fit(sample, axis=0)
    values = dist.ppf([0.01, 0.3, 0.5, 0.6, 0.99])
    assert values.shape == (5, 3)
    # Results should scale with values
    np.testing.assert_allclose(values[:, 0] * 10.0, values[:, 1])
    np.testing.assert_allclose(values[:, 1] * 10.0, values[:, 2])


def test_return_period_identity():
    dist = GumbelDistribution.fit([6.0, 5.0, 6.0, 7.0, 9.0, 5.0, 6.0, 7.0])
    values = np.linspace(4.0, 10.0, 21)
    np.testing.assert_allclose(return_period_to_value(dist, value_to_return_period(dist, values)), values)


def test_return_period_with_frequency():
    freq = np.timedelta64(24 * 60 * 60, "s")
    sample = [4.0, 3.0, 5.5, 6.0, 7.0, 5.3, 2.1]
    dist = GumbelDistribution.fit(sample, freq=freq)
    # Return periods should be scaled by the given frequency
    assert value_to_return_period(dist, 6.0).dtype == freq.dtype
