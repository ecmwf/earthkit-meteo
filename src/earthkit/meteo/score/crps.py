# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import xarray as xr

from . import array


def crps(x: xr.DataArray, y: xr.DataArray, nan_policy="propagate", ens_dim: str = "number") -> xr.DataArray:
    exclude_dims = set(y.dims) if nan_policy == "omit" else set()
    args = []
    return_dataset = False
    for arg, dim in [(x, ens_dim), (y, None)]:
        if isinstance(arg, xr.Dataset):
            arg = arg.to_dataarray().squeeze(dim="variable", drop=True)
            return_dataset = True
        if dim:
            arg = arg.transpose(dim, ...)
        args.append(arg)
    out = xr.apply_ufunc(
        array.crps,
        *args,
        input_core_dims=[x.dims for x in args],
        output_core_dims=[args[0][{ens_dim: 0}].dims],
        exclude_dims=exclude_dims,
        kwargs={"nan_policy": nan_policy},
    )
    if return_dataset:
        out.to_dataset(name=list(x.data_vars.keys())[0])
    return out
