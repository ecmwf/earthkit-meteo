# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.utils.array import array_namespace

from .. import GumbelDistribution as GumbelDistributionBase

try:
    from scipy.stats import lmoment
except ImportError:
    from ._polyfill import lmoment


def _expand_dims_after(arr, ndim, xp=None):
    if xp is None:
        xp = array_namespace(arr)
    return xp.expand_dims(xp.asarray(arr), axis=list(range(-ndim, 0)))


class GumbelDistribution(GumbelDistributionBase):

    def __init__(self, mu, sigma, freq=None):
        xp = array_namespace(mu, sigma)
        mu, sigma = xp.broadcast_arrays(xp.asarray(mu), xp.asarray(sigma))
        super().__init__(mu, sigma, freq)

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
        return super().cdf(x)

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
        return super().ppf(p)
