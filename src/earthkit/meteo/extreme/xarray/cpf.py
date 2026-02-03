# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import xarray as xr

from earthkit.meteo.utils.decorators import get_dim_from_defaults
from earthkit.meteo.utils.decorators import xarray_ufunc

from .. import array


def cpf(
    clim: xr.DataArray,
    ens: xr.DataArray,
    sort_clim: bool = True,
    sort_ens: bool = True,
    epsilon: float | None = None,
    symmetric: bool = False,
    from_zero: bool = False,
    clim_dim: str | None = None,
    ens_dim: str | None = None,
) -> xr.DataArray:
    """Compute Crossing Point Forecast (CPF).

    WARNING: this code is experimental, use at your own risk!

    Parameters
    ----------
    clim: xarray.DataArray
        Per-point climatology. The reduction dimension (quantiles) is set by ``clim_dim``.
    ens: xarray.DataArray
        Ensemble forecast. The reduction dimension (ensemble members) is set by ``ens_dim``.
    sort_clim: bool
        If True, sort the climatology first
    sort_ens: bool
        If True, sort the ensemble first
    epsilon: float or None
        If set, use this as a threshold for low-signal regions. Ignored if
        `symmetric` is True
    symmetric: bool
        If True, make CPF values below 0.5 use a symmetric computation (CPF of
        opposite values)
    from_zero: bool
        If True, start looking for a crossing from the minimum, rather than the
        median
    clim_dim: str, optional
        Name of the climatology/quantile dimension in ``clim``.
    ens_dim: str, optional
        Name of the ensemble/member dimension in ``ens``.

    Returns
    -------
    xarray.DataArray
        CPF values with the reduction dimension removed.
    """
    default_dims = ["quantiles", "percentiles", "number", "ens", "member"]
    clim_dim = get_dim_from_defaults(clim, clim_dim, default_dims)
    ens_dim = get_dim_from_defaults(ens, ens_dim, default_dims)
    if clim_dim is None or ens_dim is None:
        raise ValueError("cpf(): clim_dim and ens_dim must be provided or inferred")
    axis_clim = clim.get_axis_num(clim_dim)
    axis_ens = ens.get_axis_num(ens_dim)
    core_dims = [[clim.dims[axis_clim]], [ens.dims[axis_ens]]]
    return xarray_ufunc(
        array.cpf,
        clim,
        ens,
        sort_clim=sort_clim,
        sort_ens=sort_ens,
        epsilon=epsilon,
        symmetric=symmetric,
        from_zero=from_zero,
        clim_axis=-1,
        ens_axis=-1,
        xarray_ufunc_kwargs={
            "input_core_dims": core_dims,
            "output_core_dims": [[]],
        },
    )
