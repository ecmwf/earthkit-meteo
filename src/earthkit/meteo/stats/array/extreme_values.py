# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.utils.array import array_namespace

try:
    from scipy.stats import lmoment
except ImportError:
    from ._polyfill import lmoment


def _expand_dims_after(arr, ndim, xp=None):
    if xp is None:
        xp = array_namespace(arr)
    return xp.expand_dims(xp.asarray(arr), axis=list(range(-ndim, 0)))


class GumbelDistribution:
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
        xp = array_namespace(mu, sigma)
        self.mu, self.sigma = xp.broadcast_arrays(xp.asarray(mu), xp.asarray(sigma))
        self.freq = freq

    @classmethod
    def fit(cls, sample, axis=0, freq=None):
        """Gumbel distribution with parameters fitted to a sample of values.

        Results derived from the fitted distribution will only be meaningful
        if it is representative of the sample statistics.

        Parameters
        ----------
        sample: numpy.ndarray
            Sample values.
        axis: int
            The axis along which to compute the parameters.
        freq: None | Number | timedelta
            Temporal frequency (duration between values) of the sample.
        """
        xp = array_namespace(sample)
        lmom = lmoment(sample, axis=axis, order=[1, 2])
        sigma = lmom[1] / xp.log(2)
        mu = lmom[0] - sigma * 0.5772
        return cls(mu, sigma, freq=freq)

    @property
    def shape(self):
        """Tuple of dimensions."""
        return self.mu.shape

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.mu.ndim

    def cdf(self, x):
        """Evaluate the cumulative distribution function (CDF).

        Parameters
        ----------
        x: Number | array_like
            Input value.

        Returns
        -------
        Number | array_like
            The probability that a random variable X from the distribution is
            less than or equal to the input x.
        """
        xp = array_namespace(x, self.mu, self.sigma)
        x = _expand_dims_after(x, self.ndim, xp=xp)
        return 1.0 - xp.exp(-xp.exp((xp.asarray(self.mu) - x) / xp.asarray(self.sigma)))

    def ppf(self, p):
        """Evaluate the percent point function (PPF; inverse CDF).

        Parameters
        ----------
        p: Number | array_like
            Probability in interval [0, 1].

        Returns
        -------
        Number | array_like
            x such that the probability of a random variable from the
            distribution taking a value less than or equal to x is p.
        """
        xp = array_namespace(p, self.mu, self.sigma)
        p = _expand_dims_after(p, self.ndim, xp=xp)
        return xp.asarray(self.mu) - xp.asarray(self.sigma) * xp.log(-xp.log(1.0 - p))


def value_to_return_period(dist, value):
    """Return period of a value given a distribution of extremes.

    Use, e.g., to compute expected return periods of extreme precipitation or
    flooding events based on a timeseries of past observations.

    Parameters
    ----------
    dist: MaximumValueDistribution
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
    dist: MaximumValueDistribution
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
