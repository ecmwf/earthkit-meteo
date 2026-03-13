# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import numbers

from ...utils.decorators import xarray_ufunc
from .. import array

__all__ = ["fit_gumbel", "value_to_return_period", "return_period_to_value"]


def _ensure_compat(x, forbidden_dims=None):
    if isinstance(x, numbers.Number):
        import xarray as xr

        x = xr.DataArray(x)
    if forbidden_dims is not None:
        forbidden_dims = set(forbidden_dims)
        for dim in x.dims:
            if dim in forbidden_dims:
                raise ValueError(f"shared dimension '{dim}' on input and distribution not allowed")
    return x


def fit_gumbel(sample, dim):
    """Gumbel distribution with parameters fitted to a sample of values.

    .. warning:: Experimental API. This function may change or be removed without notice.

    Results derived from the fitted distribution will only be meaningful
    if it is representative of the sample statistics.

    Parameters
    ----------
    sample: xarray.DataArray
        Sample values.
    dim: str
        Dimension name over which to compute the parameters.

    Returns
    -------
    GumbelDistribution
        Fitting over a dimension of a multi-dimensional sample array, the
        outcome is a collection of (scalar-valued) distributions.
    """
    if dim not in sample.dims:
        raise ValueError(f"cannot fit over dimension '{dim}' with sample dimensions {sample.dims}")
    axis = sample.dims.index(dim)
    parameter_dims = [d for d in sample.dims if d != dim]
    parameter_coords = {dim: values for dim, values in sample.coords.items() if dim in parameter_dims}
    return array.fit_gumbel(sample.data, dim=axis, dims=parameter_dims, coords=parameter_coords)


def value_to_return_period(value, dist):
    """Return period of a value given a distribution of extremes.

    .. warning:: Experimental API. This function may change or be removed without notice.

    Parameters
    ----------
    value: number | xarray.DataArray
        Input value(s).
    dist: earthkit.meteo.stats.array.GumbelDistribution
        Probability distribution.

    Returns
    -------
    xarray.DataArray
        The return period of the input value. Distribution dimensions are added
        at the end.
    """
    value = _ensure_compat(value, forbidden_dims=dist.dims)
    return (
        xarray_ufunc(
            array.value_to_return_period,
            value,
            dist=dist,
            xarray_ufunc_kwargs={
                "input_core_dims": [[]],
                "output_core_dims": [dist.dims],
            },
        )
        .assign_coords(dist.coords)
        .rename("return_period")
    )
    # TODO assign metadata attributs (unit, etc.), but where from?


def return_period_to_value(return_period, dist):
    """Value for a given return period of a distribution of extremes.

    .. warning:: Experimental API. This function may change or be removed without notice.

    Parameters
    ----------
    return_period: number | xarray.DataArray
        Input return period.
    dist: earthkit.meteo.stats.array.GumbelDistribution
        Probability distribution.

    Returns
    -------
    xarray.DataArray
        Value with return period equal to the input return period. Distribution
        dimensions are added at the end.
    """
    return_period = _ensure_compat(return_period, forbidden_dims=dist.dims)
    return xarray_ufunc(
        array.return_period_to_value,
        return_period,
        dist=dist,
        xarray_ufunc_kwargs={"input_core_dims": [[]], "output_core_dims": [dist.dims]},
    ).assign_coords(dist.coords)
    # TODO assign a name and metadata attributes (unit, etc.), but where from?
