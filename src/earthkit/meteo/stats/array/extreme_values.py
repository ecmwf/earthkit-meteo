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
    mu: array_like
        Offset parameter.
    sigma: array_like
        Scale parameter.
    freq: None | Number | timedelta
        Temporal frequency (duration between values, technically the inverse
        frequency) of the represented data. Provides additional context, e.g.,
        to scale return periods computed from the distribution.
    """

    def __init__(self, mu, sigma, dims=None, coords=None):
        xp = array_namespace(mu, sigma)
        self.mu, self.sigma = xp.broadcast_arrays(mu, sigma)
        # Optional metadata
        self._dims = dims
        self._coords = coords
        assert self._dims is None or len(self._dims) == self.ndim

    @property
    def shape(self):
        """Tuple of dimensions."""
        return self.mu.shape

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.mu.ndim

    @property
    def dims(self):
        if self._dims is not None:
            return tuple(self._dims)
        return tuple(f"dim{i}" for i in self.ndim)

    @property
    def coords(self):
        return self._coords

    def cdf(self, x):
        """Evaluate the cumulative distribution function (CDF).

        Parameters
        ----------
        x: array_like
            Input value.

        Returns
        -------
        array_like
            The probability that a random variable X from the distribution is
            less than or equal to the input x.
        """
        xp = array_namespace(x, self.mu, self.sigma)
        x = _expand_dims_after(x, self.ndim, xp=xp)
        return 1.0 - xp.exp(-xp.exp((self.mu - x) / self.sigma))

    def ppf(self, p):
        """Evaluate the percent point function (PPF; inverse CDF).

        Parameters
        ----------
        p: array_like
            Probability in interval [0, 1].

        Returns
        -------
        array_like
            x such that the probability of a random variable from the
            distribution taking a value less than or equal to x is p.
        """
        xp = array_namespace(p, self.mu, self.sigma)
        p = _expand_dims_after(p, self.ndim, xp=xp)
        return self.mu - self.sigma * xp.log(-xp.log(1.0 - p))


def fit_gumbel(sample, over=0, **kwargs):
    """Gumbel distribution with parameters fitted to a sample of values.

    Results derived from the fitted distribution will only be meaningful
    if it is representative of the sample statistics.

    Parameters
    ----------
    sample: numpy.ndarray
        Sample values.
    over: int
        The axis along which to compute the distribution parameters.
    **kwargs: dict[str,Any]
        Keyword arguments forwarded to the distribution constructor.
    """
    xp = array_namespace(sample)
    lmom = lmoment(sample, axis=over, order=[1, 2])
    sigma = lmom[1] / xp.log(2)
    mu = lmom[0] - sigma * 0.5772
    return GumbelDistribution(mu, sigma, **kwargs)


def value_to_return_period(value, dist):
    """Return period of a value given a distribution of extremes.

    Use, e.g., to compute expected return periods of extreme precipitation or
    flooding events based on a timeseries of past observations.

    Parameters
    ----------
    dist: GumbelDistribution
        Probability distribution.
    value: array_like
        Input value(s).

    Returns
    -------
    array_like
        The return period of the input value, scaled with the frequency
        information of the distribution if attached.
    """
    xp = array_namespace(value)
    return 1.0 / dist.cdf(xp.asarray(value))


def return_period_to_value(return_period, dist):
    """Value for a given return period of a distribution of extremes.

    Parameters
    ----------
    dist: GumbelDistribution
        Probability distribution.
    return_period: array_like
        Input return period. Must be compatible with the frequency information
        of the distribution if attached.

    Returns
    -------
    array_like
        Value with return period equal to the input return period.
    """
    xp = array_namespace(return_period)
    return dist.ppf(1.0 / xp.asarray(return_period))
