# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import TYPE_CHECKING
from typing import Any
from typing import TypeAlias
from typing import overload

from ..utils.decorators import dispatch

ArrayLike: TypeAlias = Any

if TYPE_CHECKING:
    import xarray  # type: ignore[import]

    from .array.extreme_values import GumbelDistribution


@overload
def fit_gumbel(sample: "ArrayLike", dim: int) -> "GumbelDistribution": ...


@overload
def fit_gumbel(sample: "xarray.DataArray", dim: str) -> "GumbelDistribution": ...


def fit_gumbel(sample, dim):
    """Gumbel distribution with parameters fitted to a sample of values.

    .. warning:: Experimental API. This function may change or be removed without notice.

    Results derived from the fitted distribution will only be meaningful
    if it is representative of the sample statistics.

    Parameters
    ----------
    sample: xarray.DataArray
        Sample values.
    dim: str or int
        Dimension name (for xarray) or axis index (for array-like) over which to
        compute the parameters.

    Returns
    -------
    GumbelDistribution
        Fitting over a dimension of a multi-dimensional sample array, the
        outcome is a collection of (scalar-valued) distributions.

    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(fit_gumbel, xarray=True, array=True)
    return dispatched(sample, dim=dim)


@overload
def value_to_return_period(value: "ArrayLike", dist: "GumbelDistribution") -> "ArrayLike": ...


@overload
def value_to_return_period(value: "xarray.DataArray", dist: "GumbelDistribution") -> "xarray.DataArray": ...


def value_to_return_period(value, dist):
    """Return period of a value given a distribution of extremes.

    .. warning:: Experimental API. This function may change or be removed without notice.

    Use, e.g., to compute expected return periods of extreme precipitation or
    flooding events based on a timeseries of past observations.

    Parameters
    ----------
    value: xarray.DataArray
        Input value(s).
    dist: GumbelDistribution
        Probability distribution.

    Returns
    -------
    xarray.DataArray
        The return period of the input value. Distribution dimensions are added
        at the end.
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(value_to_return_period, xarray=True, array=True)
    return dispatched(value, dist)


@overload
def return_period_to_value(return_period: "ArrayLike", dist: "GumbelDistribution") -> "ArrayLike": ...


@overload
def return_period_to_value(
    return_period: "xarray.DataArray",
    dist: "GumbelDistribution",
) -> "xarray.DataArray": ...


def return_period_to_value(return_period, dist):
    """Value for a given return period of a distribution of extremes.

    .. warning:: Experimental API. This function may change or be removed without notice.

    Parameters
    ----------
    return_period: xarray.DataArray
        Input return period.
    dist: GumbelDistribution
        Probability distribution.

    Returns
    -------
    xarray.DataArray
        Value with return period equal to the input return period. Distribution
        dimensions are added at the end.
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(return_period_to_value, xarray=True, array=True)
    return dispatched(return_period, dist)
