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

from earthkit.utils.array import array_namespace


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
    if clim_dim == ens_dim:
        clim_dim = f"clim_{ens_dim}"
        clim = clim.rename({ens_dim: clim_dim})
        
    xp = array_namespace(clim.data, ens.data)

    # locate missing values
    missing_mask = xr.ufuncs.logical_or(xr.ufuncs.isnan(clim).sum(dim=clim_dim), xr.ufuncs.isnan(ens).sum(dim=ens_dim))

    # Compute fraction of the forecast below climatology
    nclim, npoints = clim.shape
    nens, npoints_ens = ens.shape
    assert npoints == npoints_ens
    frac = clim.groupby(clim_dim).map(lambda x: (ens <= x).sum(dim=ens_dim) / nens)

    # Compute formula coefficients
    p = xp.linspace(0.0, 1.0, nclim)
    dp = 1 / (nclim - 1)
    dFdp = frac.diff(dim=clim_dim, label="lower") / dp

    acosdiff = xp.diff(xp.arccos(xp.sqrt(p)))
    proddiff = xp.diff(xp.sqrt(p * (1.0 - p)))

    acoef = (1.0 - 2.0 * p[:-1]) * acosdiff + proddiff

    ds = xr.Dataset({
        "frac": frac[{clim_dim: slice(0, nclim - 1)}], 
        "acosdiff": xr.DataArray(acosdiff, dims=clim_dim), 
        "acoef": xr.DataArray(acoef, dims=clim_dim), 
        "dFdp": dFdp, 
        "proddiff": xr.DataArray(proddiff, dims=clim_dim)
    })
    mapped_ds = (
        ds.groupby(clim_dim)
        .map(lambda ds: (2.0 * ds.frac - 1.0) * ds.acosdiff + ds.acoef * ds.dFdp - ds.proddiff)
    )

    # compute EFI from coefficients
    efi = xr.zeros_like(ens[{ens_dim: 0}])
    ##################################
    if eps > 0:
        efimax = xp.zeros(npoints)
        ds["mask"] = clim[{clim_dim: slice(1, nclim)}] > eps
        efi = xr.where(ds.mask, mapped_ds, 0.0).sum(clim_dim)
        efimax = (
            ds.groupby(clim_dim)
            .map(lambda ds: xr.where(ds.mask, -ds.acosdiff - ds.proddiff, 0.0))
            .sum(clim_dim)
        )
        efimax = xr.ufuncs.fmax(efimax, xp.asarray(eps))
        efi /= efimax
    else:
        efi = (2.0 / np.pi) * mapped_ds.sum(clim_dim)
    ##################################

    # apply missing values
    efi[missing_mask] = xp.nan

    return efi