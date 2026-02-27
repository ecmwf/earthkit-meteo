# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

from ...utils.decorators import xarray_ufunc
from .. import array

__all__ = ["fit_gumbel", "value_to_return_period", "return_period_to_value"]


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
    assert over in sample.dims
    over_axis = sample.dims.index(over)
    parameter_dims = [dim for dim in sample.dims if dim != over]
    parameter_coords = {dim: values for dim, values in sample.coords.items() if dim in parameter_dims}
    return array.fit_gumbel(sample.data, over=over_axis, dims=parameter_dims, coords=parameter_coords)


def value_to_return_period(value, dist):
    """Return period of a value given a distribution of extremes.

    .. warning:: Experimental API. This function may change or be removed without notice.

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
    assert dist.dims is None or not (set(value.dims) & set(dist.dims))
    return xarray_ufunc(
        array.value_to_return_period,
        value,
        dist=dist,
        xarray_ufunc_kwargs={"input_core_dims": [[]], "output_core_dims": [dist.dims]},
    ).assign_coords(dist.coords)


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
    assert dist.dims is None or not (set(return_period.dims) & set(dist.dims))
    return xarray_ufunc(
        array.return_period_to_value,
        return_period,
        dist=dist,
        xarray_ufunc_kwargs={"input_core_dims": [[]], "output_core_dims": [dist.dims]},
    ).assign_coords(dist.coords)
