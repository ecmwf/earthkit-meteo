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


def efi(
    clim: xr.DataArray,
    ens: xr.DataArray,
    eps: float = -0.1,
    clim_dim: str | None = None,
    ens_dim: str | None = None,
) -> xr.DataArray:
    """Compute Extreme Forecast Index (EFI).

    Parameters
    ----------
    clim: xarray.DataArray
        Sorted per-point climatology. The reduction dimension (quantiles) is set by ``clim_dim``.
    ens: xarray.DataArray
        Ensemble forecast. The reduction dimension (ensemble members) is set by ``ens_dim``.
    eps: (float)
        Epsilon factor for zero values
    clim_dim: str, optional
        Name of the climatology/quantile dimension in ``clim``.
    ens_dim: str, optional
        Name of the ensemble/member dimension in ``ens``.

    Returns
    -------
    xarray.DataArray
        EFI values.
    """
    default_dims = ["quantiles", "percentiles", "number", "ens", "member"]
    clim_dim = get_dim_from_defaults(clim, clim_dim, default_dims)
    ens_dim = get_dim_from_defaults(ens, ens_dim, default_dims)
    if clim_dim is None or ens_dim is None:
        raise ValueError("efi(): clim_dim and ens_dim must be provided or inferred")
    core_dims = [[clim_dim], [ens_dim]]
    return xarray_ufunc(
        array.efi,
        clim,
        ens,
        eps=eps,
        clim_axis=-1,
        ens_axis=-1,
        xarray_ufunc_kwargs={
            "input_core_dims": core_dims,
            "output_core_dims": [[]],
        },
    )
