# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import xarray as xr
import numpy as np

from earthkit.meteo.extreme import array


def efi(clim: xr.DataArray, ens: xr.DataArray, eps=-0.1, ens_dim: str = "number", clim_dim: str = "quantile") -> xr.DataArray:
    """Compute Extreme Forecast Index (EFI)

    Parameters
    ----------
    clim: array-like (nclim, npoints)
        Sorted per-point climatology
    ens: array-like (nens, npoints)
        Ensemble forecast
    eps: (float)
        Epsilon factor for zero values

    Returns
    -------
    array-like (npoints)
        EFI values
    """

    return xr.apply_ufunc(
        array.efi, 
        clim.transpose(clim_dim, ...), 
        ens.transpose(ens_dim, ...), 
        input_core_dims=[clim.dims, ens.dims],
        output_core_dims=[ens[{ens_dim: 0}].dims],
        kwargs={"eps": eps}
    )