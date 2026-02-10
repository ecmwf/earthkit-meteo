# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

# import numba
# from numba import float64, float32

from earthkit.utils.array import array_namespace

from .utils import (
    flatten_extreme_input,
    validate_extreme_shapes,
)


def efi(clim, ens, eps=-0.1, clim_axis=0, ens_axis=0):
    """Compute Extreme Forecast Index (EFI).

    Parameters
    ----------
    clim: array-like
        Sorted per-point climatology. The reduction axis (quantiles) is set by ``clim_axis``.
    ens: array-like
        Ensemble forecast. The reduction axis (ensemble members) is set by ``ens_axis``.
    eps: (float)
        Epsilon factor for zero values
    clim_axis: int
        Axis index of the climatology/quantile dimension in ``clim``. Default is 0.
    ens_axis: int
        Axis index of the ensemble/member dimension in ``ens``. Default is 0.

    Returns
    -------
    array-like
        EFI values.
    """

    xp = array_namespace(clim, ens)
    clim = xp.asarray(clim)
    ens = xp.asarray(ens)
    device = xp.device(clim)
    validate_extreme_shapes(
        func="efi",
        clim_shape=clim.shape,
        ens_shape=ens.shape,
        clim_axis=clim_axis,
        ens_axis=ens_axis,
    )
    # Compute fraction of the forecast below climatology
    clim, out_shape = flatten_extreme_input(xp, clim, clim_axis)
    ens, _ = flatten_extreme_input(xp, ens, ens_axis)

    nclim = clim.shape[0]
    nens = ens.shape[0]
    npoints = clim.shape[1]

    # locate missing values
    missing_mask = xp.logical_or(
        xp.sum(xp.isnan(clim), axis=0), xp.sum(xp.isnan(ens), axis=0)
    )

    frac = xp.zeros_like(clim)
    ##################################
    for icl in range(nclim):
        frac[icl, :] = xp.sum(ens[:, :] <= clim[icl, xp.newaxis, :], axis=0)
    ##################################
    frac /= nens

    # Compute formula coefficients
    p = xp.linspace(0.0, 1.0, nclim)
    dFdp = xp.diff(frac, axis=0) * (nclim - 1)

    acosdiff = xp.diff(xp.acos(xp.sqrt(p)))
    proddiff = xp.diff(xp.sqrt(p * (1.0 - p)))

    acoef = (1.0 - 2.0 * p[:-1]) * acosdiff + proddiff

    # compute EFI from coefficients
    efi = xp.zeros(npoints, device=device)
    ##################################
    if eps > 0:
        efimax = xp.zeros(npoints, device=device)
        for icl in range(nclim - 1):
            mask = clim[icl + 1, :] > eps
            dEFI = xp.where(
                mask,
                (2.0 * frac[icl, :] - 1.0) * acosdiff[icl]
                + acoef[icl] * dFdp[icl, :]
                - proddiff[icl],
                0.0,
            )
            defimax = xp.where(mask, -acosdiff[icl] - proddiff[icl], 0.0)
            efi += dEFI
            efimax += defimax
        efimax = xp.maximum(efimax, eps)
        efi /= efimax
    else:
        for icl in range(nclim - 1):
            dEFI = (
                (2.0 * frac[icl, :] - 1.0) * acosdiff[icl]
                + acoef[icl] * dFdp[icl, :]
                - proddiff[icl]
            )
            efi += dEFI
        efi *= 2.0 / xp.pi
    ##################################

    # apply missing values
    efi[missing_mask] = xp.nan
    return xp.reshape(efi, out_shape)
