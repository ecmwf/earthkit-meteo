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


def efi(clim: xr.DataArray, ens: xr.DataArray, eps=-0.1, clim_dim: str = "quantile", ens_dim: str = "number") -> xr.DataArray:
    return xr.apply_ufunc(
        array.efi, 
        clim.transpose(clim_dim, ...), 
        ens.transpose(ens_dim, ...), 
        input_core_dims=[clim.dims, ens.dims],
        output_core_dims=[ens[{ens_dim: 0}].dims],
        kwargs={"eps": eps}
    )