# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import xarray as xr

from . import array  # noqa


def cpf(
    clim: xr.DataArray,
    ens: xr.DataArray,
    sort_clim: bool = True,
    sort_ens: bool = True,
    epsilon: bool = None,
    symmetric: bool = False,
    ens_dim: str = "number",
    clim_dim: str = "quantile",
) -> xr.DataArray:
    return xr.apply_ufunc(
        array.cpf,
        clim.transpose(clim_dim, ...),
        ens.transpose(ens_dim, ...),
        input_core_dims=[clim.dims, ens.dims],
        output_core_dims=[ens[{ens_dim: 0}].dims],
        kwargs={
            "sort_clim": sort_clim,
            "sort_ens": sort_ens,
            "epsilon": epsilon,
            "symmetric": symmetric,
        },
    )
