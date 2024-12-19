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
        from scipy.stats import lmoment

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
