# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from ..utils.decorators import dispatch


def fit_gumbel(sample, over):
    """Gumbel distribution with parameters fitted to a sample of values.

    .. warning:: Experimental API. This function may change or be removed without notice.

    Results derived from the fitted distribution will only be meaningful
    if it is representative of the sample statistics.

    Parameters
    ----------
    sample: xarray.DataArray
        Sample values.
    over: str
        The dimension over which to compute the parameters.

    Returns
    -------
    GumbelDistribution
        Fitting over a dimension of a multi-dimensional sample array, the
        outcome is a collection of (scalar-valued) distributions.

    """
    return dispatch(fit_gumbel, sample, over)


def value_to_return_period(value, dist):
    """Return period of a value given a distribution of extremes.

    .. warning:: Experimental API. This function may change or be removed without notice.

    Use, e.g., to compute expected return periods of extreme precipitation or
    flooding events based on a timeseries of past observations.

    Parameters
    ----------
    dist: GumbelDistribution
        Probability distribution.
    value: xarray.DataArray
        Input value(s).

    Returns
    -------
    xarray.DataArray
        The return period of the input value. Distribution dimensions are added
        at the end.
    """
    return dispatch(value_to_return_period, value, dist)


def return_period_to_value(return_period, dist):
    """Value for a given return period of a distribution of extremes.

    .. warning:: Experimental API. This function may change or be removed without notice.

    Parameters
    ----------
    dist: GumbelDistribution
        Probability distribution.
    return_period: xarray.DataArray
        Input return period.

    Returns
    -------
    xarray.DataArray
        Value with return period equal to the input return period. Distribution
        dimensions are added at the end.
    """
    return dispatch(return_period_to_value, return_period, dist)
