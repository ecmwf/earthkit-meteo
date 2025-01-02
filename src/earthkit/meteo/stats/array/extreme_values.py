# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from .distributions import MaxGumbel


class MaximumStatistics:
    """Recurrence statistics for a sample of maximum values.

    All statistics are computed from a fitted continuous probability
    distribution. Results will only be meaningful if this fitted distribution
    is representative of the sample statistics.

    Use, e.g., to compute expected return periods of extreme precipitation or
    flooding events based on past observations.

    Parameters
    ----------
    sample: array_like
        Input maximum values. Samples must be representative of intervals
        of equal length (freq) along the axis of computation (axis).
    axis: int
        The axis along which to compute the statistics.
    freq: number | timedelta
        Temporal frequency of the input data. Used to scale return periods.
        Defaults to 1, i.e., no scaling applied. Note: when supplying a numpy
        timedelta64, the unit carries over to return periods.
    dist: ContinuousDistribution
        Continuous probability distribution fitted to the input data.
    """

    def __init__(self, sample, axis=0, freq=1.0, dist=MaxGumbel):
        self._sample = sample
        self._dist = dist.fit(sample, axis=axis)
        self._freq = freq

    @property
    def dist(self):
        """Estimated continuous probability distribution for the data."""
        return self._dist

    @property
    def freq(self):
        """Temporal frequency used for scaling return periods."""
        return self._freq

    def probability_of_threshold(self, threshold):
        """Probability of threshold exceedance.

        Parameters
        ----------
        threshold: Number | array_like
            Input threshold.

        Returns
        -------
        array_like
            The probability ([0, 1]) of a value to exceed the input threshold
            in a time interval.
        """
        return self.dist.cdf(threshold)

    def return_period_of_threshold(self, threshold):
        """Return period of threshold exceedance.

        Parameters
        ----------
        threshold: Number | array_like
            Input threshold.

        Returns
        -------
        array_like
            The return period of the input threshold.
        """
        return self.freq / self.probability_of_threshold(threshold)

    def threshold_of_probability(self, probability):
        """Threshold of a given probability of exceedance.

        Parameters
        ----------
        probability: Number | array_like
            Input probability.

        Returns
        -------
        array_like
            Threshold with exceedance probability equal to the input
            probability.
        """
        return self.dist.ppf(probability)

    def threshold_of_return_period(self, return_period):
        """Threshold of a given return period.

        Parameters
        ----------
        return_period: Number | array_like
            Input return period.

        Returns
        -------
        array_like
            Threshold with return period equal to the input return period.
        """
        return self.threshold_of_probability(self.freq / return_period)
