# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import abc

from earthkit.utils.array import array_namespace

from ..utils.decorators import dispatch


class GumbelDistribution(abc.ABC):
    """Gumbel distribution for extreme value statistics.

    Parameters
    ----------
    mu: Number | array_like
        Offset parameter.
    sigma: Number | array_like
        Scale parameter.
    freq: None | Number | timedelta
        Temporal frequency (duration between values, technically the inverse
        frequency) of the represented data. Provides additional context, e.g.,
        to scale return periods computed from the distribution.
    """

    def __init__(self, mu, sigma, freq=None):
        self.mu = mu
        self.sigma = sigma
        self.freq = freq

    @classmethod
    @abc.abstractmethod
    def fit(cls, sample, **kwargs):
        """Fit a distribution with parameters to a sample of values.

        Results derived from the fitted distribution will only be meaningful
        if it is representative of the sample statistics.

        Parameters
        ----------
        sample: numpy.ndarray
            Sample values.
        **kwargs:
            Additional keyword arguments supplied to the distribution.
        """
        return dispatch(cls.fit, sample, **kwargs)

    @property
    def shape(self):
        """Tuple of dimensions."""
        return self.mu.shape

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.mu.ndim

    def cdf(self, x):
        xp = array_namespace(x, self.mu, self.sigma)
        return 1.0 - xp.exp(-xp.exp((self.mu - x) / self.sigma))

    def ppf(self, p):
        xp = array_namespace(p, self.mu, self.sigma)
        return self.mu - self.sigma * xp.log(-xp.log(1.0 - p))


def value_to_return_period(dist, value):
    """Return period of a value given a distribution of extremes.

    Use, e.g., to compute expected return periods of extreme precipitation or
    flooding events based on a timeseries of past observations.

    Parameters
    ----------
    dist: GumbelDistribution
        Probability distribution.
    value: Number | array_like
        Input value(s).

    Returns
    -------
    array_like
        The return period of the input value, scaled with the frequency
        information of the distribution if attached.
    """
    freq = 1.0 if dist.freq is None else dist.freq
    return freq / dist.cdf(value)


def return_period_to_value(dist, return_period):
    """Value for a given return period of a distribution of extremes.

    Parameters
    ----------
    dist: GumbelDistribution
        Probability distribution.
    return_period: Number | array_like
        Input return period. Must be compatible with the frequency information
        of the distribution if attached.
    freq: number | timedelta
        Temporal frequency of the input dataUsed to scale return periods.

    Returns
    -------
    array_like
        Value with return period equal to the input return period.
    """
    freq = 1.0 if dist.freq is None else dist.freq
    return dist.ppf(freq / return_period)
