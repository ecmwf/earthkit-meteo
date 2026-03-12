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

    .. warning:: Experimental API. This class may change or be removed without notice.

    Parameters
    ----------
    mu: array_like
        Offset parameter.
    sigma: array_like
        Scale parameter.
    dims: tuple[str], optional
        Ordered sequence of dimension labels. To be used by metadata-aware
        functions working with the distribution.
    coords: dict[str,Any], optional
        Coordinates corresponding to the labelled dimensions provided in `dims`.
        To be used by metadata-aware functions working with the distribution.
    """

    def __init__(self, mu, sigma, dims=None, coords=None):
        xp = array_namespace(mu, sigma)
        self.mu, self.sigma = xp.broadcast_arrays(mu, sigma)
        # Optional metadata
        self._dims = tuple(dims) if dims is not None else dims
        self._coords = coords
        assert self.dims is None or len(self.dims) == self.ndim

    @property
    def shape(self) -> tuple[int]:
        """Tuple of dimensions."""
        return self.mu.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.mu.ndim

    @property
    def dims(self) -> None | tuple[str]:
        return self._dims

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

    .. warning:: Experimental API. This function may change or be removed without notice.

    Results derived from the fitted distribution will only be meaningful
    if it is representative of the sample statistics.

    Fitting over axes other than 0 is only possible if scipy is installed.

    Parameters
    ----------
    sample: numpy.ndarray
        Sample values.
    over: int
        The axis along which to compute the distribution parameters.
    **kwargs: dict[str,Any]
        Keyword arguments forwarded to the distribution constructor.

    Returns
    -------
    GumbelDistribution
        Fitting over a dimension of a multi-dimensional sample array, the
        outcome is a collection of (scalar-valued) distributions.
    """
    xp = array_namespace(sample)
    lmom = lmoment(sample, axis=over, order=[1, 2])
    sigma = lmom[1] / xp.log(2)
    mu = lmom[0] - sigma * 0.5772
    return GumbelDistribution(mu, sigma, **kwargs)


def value_to_return_period(value, dist):
    """Return period of a value given a distribution of extremes.

    .. warning:: Experimental API. This function may change or be removed without notice.

    Use, e.g., to compute expected return periods of extreme precipitation or
    flooding events based on a timeseries of past observations.

    Parameters
    ----------
    value: array_like
        Input value(s).
    dist: GumbelDistribution
        Probability distribution.

    Returns
    -------
    array_like
        The return period of the input value.
    """
    xp = array_namespace(value)
    return 1.0 / dist.cdf(xp.asarray(value))


def return_period_to_value(return_period, dist):
    """Value for a given return period of a distribution of extremes.

    .. warning:: Experimental API. This function may change or be removed without notice.

    Parameters
    ----------
    return_period: array_like
        Input return period.
    dist: GumbelDistribution
        Probability distribution.

    Returns
    -------
    array_like
        Value with return period equal to the input return period.
    """
    xp = array_namespace(return_period)
    return dist.ppf(1.0 / xp.asarray(return_period))
