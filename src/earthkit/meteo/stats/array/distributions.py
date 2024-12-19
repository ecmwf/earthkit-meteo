# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import abc

import numpy as np


class ContinuousDistribution(abc.ABC):
    """Continuous probabiliy distribution function

    Partially implements the interface of scipy.stats.rv_continuous, but all
    methods should be applicable along an axis so fields can be processed in
    a grid point-wise fashion.
    """

    @abc.abstracmethod
    @classmethod
    def fit(cls, sample, axis):
        """Determine distribution parameters from a sample of data"""

    @abc.abstractmethod
    def cdf(self, x):
        """Evaluate the continuous distribution function"""

    @abc.abstractmethod
    def ppf(self, x):
        """Evaluate the inverse CDF (percent point function)"""


# Temporary drop-in replacement for scipy.stats.lmoment from scipy v0.15
def _lmoment(sample, order=(1, 2), axis=0):
    """Compute first 2 L-moments of dataset along first axis"""
    if len(order) != 2 or order[0] != 1 or order[1] != 2:
        raise NotImplementedError
    if axis != 0:
        raise NotImplementedError

    nmoments = 3

    # At least four values needed to make a sample L-moments estimation
    nvalues, *rest_shape = sample.shape
    if nvalues < 4:
        raise ValueError("Insufficient number of values to perform sample L-moments estimation")

    sample = np.sort(sample, axis=0)  # ascending order

    sums = np.zeros_like(sample, shape=(nmoments, *rest_shape))

    for i in range(1, nvalues + 1):
        z = i
        term = sample[i - 1]
        sums[0] = sums[0] + term
        for j in range(1, nmoments):
            z -= 1
            term = term * z
            sums[j] = sums[j] + term

    y = float(nvalues)
    z = float(nvalues)
    sums[0] = sums[0] / z
    for j in range(1, nmoments):
        y = y - 1.0
        z = z * y
        sums[j] = sums[j] / z

    k = nmoments
    p0 = -1.0
    for _ in range(2):
        ak = float(k)
        p0 = -p0
        p = p0
        temp = p * sums[0]
        for i in range(1, k):
            ai = i
            p = -p * (ak + ai - 1.0) * (ak - ai) / (ai * ai)
            temp = temp + (p * sums[i])
        sums[k - 1] = temp
        k = k - 1

    return np.stack([sums[0], sums[1]])


class MaxGumbel(ContinuousDistribution):
    """Gumbel distribution for extreme values

    Parameters
    ----------
    mu: array_like
        Offset parameter.
    sigma: array_like
        Scale parameter.
    """

    def __init__(self, mu, sigma):  # TODO: needs an axis argument?
        self.mu = np.asarray(mu)
        self.sigma = np.asarray(sigma)
        assert self.mu.shape == self.sigma.shape

    @classmethod
    def fit(cls, sample, axis=0):
        """Gumbel distribution with parameters fitted to sample values

        Parameters
        ----------
        sample: array_like
            Sample values.
        axis: int
            The axis along which to compute the parameters.
        """
        try:
            from scipy.stats import lmoment
        except ImportError:
            lmoment = _lmoment

        lmom = lmoment(sample, axis=axis, order=[1, 2])
        sigma = lmom[1] / np.log(2)
        mu = lmom[0] - sigma * 0.5772
        return cls(mu, sigma)

    def cdf(self, x):
        """Evaluate the cumulative distribution function

        Parameters
        ----------
        x: array_like
            Input value(s).

        Returns
        -------
        The probability that a random variable X from the distribution is less
        than or equal to the input x.
        """
        return 1.0 - np.exp(-np.exp((self.mu - x) / self.sigma))  # TODO vectorize along axis properly

    def ppf(self, p):
        """Evaluate the inverse cumulative distribution function

        Parameters
        ----------
        p: array_like
            Probability in interval [0, 1].

        Returns
        -------
        x such that the probability of a random variable from the distribution
        taking a value less than or equal to x is p.
        """
        return self.mu - self.sigma * np.log(-np.log(1.0 - p))  # TODO vectorize along axis properly
