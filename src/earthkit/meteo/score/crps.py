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
    return xr.apply_ufunc(
        array.crps,
        x.transpose(ens_dim, ...),
        y,
        input_core_dims=[x.dims, y.dims],
        output_core_dims=[x[{ens_dim: 0}].dims],
        exclude_dims=exclude_dims,
        kwargs={"nan_policy": nan_policy},
    )
