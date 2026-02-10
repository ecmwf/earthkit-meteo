# (C) Copyright 2026 ECMWF.
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


def sot(
    clim: xr.DataArray,
    ens: xr.DataArray,
    perc: int,
    eps: float = -1e4,
    clim_dim: str | None = None,
    ens_dim: str | None = None,
) -> xr.DataArray:
    """Compute Shift of Tails (SOT)
    from climatology percentiles (sorted)
    and ensemble forecast (not sorted)

    Parameters
    ----------
    clim: xarray.DataArray
        Model climatology (percentiles). The reduction dimension is set by ``clim_dim``.
    ens: xarray.DataArray
        Ensemble forecast. The reduction dimension is set by ``ens_dim``.
    perc: int
        Percentile value (typically 10 or 90)
    eps: (float)
        Epsilon factor for zero values
    clim_dim: str, optional
        Name of the climatology/quantile dimension in ``clim``.
    ens_dim: str, optional
        Name of the ensemble/member dimension in ``ens``.

    Returns
    -------
    xarray.DataArray
        SOT values.
    """
    default_dims = ["quantiles", "percentiles", "number", "ens", "member"]
    clim_dim = get_dim_from_defaults(clim, clim_dim, default_dims)
    ens_dim = get_dim_from_defaults(ens, ens_dim, default_dims)
    if clim_dim is None or ens_dim is None:
        raise ValueError("sot(): clim_dim and ens_dim must be provided or inferred")
    axis_clim = clim.get_axis_num(clim_dim)
    axis_ens = ens.get_axis_num(ens_dim)
    core_dims = [[clim.dims[axis_clim]], [ens.dims[axis_ens]]]
    return xarray_ufunc(
        array.sot,
        clim,
        ens,
        perc=perc,
        eps=eps,
        clim_axis=-1,
        ens_axis=-1,
        xarray_ufunc_kwargs={
            "input_core_dims": core_dims,
            "output_core_dims": [[]],
        },
    )


def sot_unsorted(
    clim: xr.DataArray,
    ens: xr.DataArray,
    perc: int,
    eps: float = -1e4,
    clim_dim: str | None = None,
    ens_dim: str | None = None,
) -> xr.DataArray:
    """Compute Shift of Tails (SOT)
    from climatology percentiles (sorted)
    and ensemble forecast (not sorted)

    Parameters
    ----------
    clim: xarray.DataArray
        Model climatology (percentiles). The reduction dimension is set by ``clim_dim``.
    ens: xarray.DataArray
        Ensemble forecast. The reduction dimension is set by ``ens_dim``.
    perc: int
        Percentile value (typically 10 or 90)
    eps: (float)
        Epsilon factor for zero values
    clim_dim: str, optional
        Name of the climatology/quantile dimension in ``clim``.
    ens_dim: str, optional
        Name of the ensemble/member dimension in ``ens``.

    Returns
    -------
    xarray.DataArray
        SOT values.
    """
    default_dims = ["quantiles", "percentiles", "number", "ens", "member"]
    clim_dim = get_dim_from_defaults(clim, clim_dim, default_dims)
    ens_dim = get_dim_from_defaults(ens, ens_dim, default_dims)
    if clim_dim is None or ens_dim is None:
        raise ValueError("sot_unsorted(): clim_dim and ens_dim must be provided or inferred")
    core_dims = [[clim_dim], [ens_dim]]
    return xarray_ufunc(
        array.sot_unsorted,
        clim,
        ens,
        perc=perc,
        eps=eps,
        clim_axis=-1,
        ens_axis=-1,
        xarray_ufunc_kwargs={
            "input_core_dims": core_dims,
            "output_core_dims": [[]],
        },
    )
