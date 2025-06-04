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


def efi(clim, ens, eps=-0.1):
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

    xp = array_namespace(clim, ens)
    clim = xp.asarray(clim)
    ens = xp.asarray(ens)

    # locate missing values
    missing_mask = xp.logical_or(xp.sum(xp.isnan(clim), axis=0), xp.sum(xp.isnan(ens), axis=0))

    # Compute fraction of the forecast below climatology
    nclim, npoints = clim.shape
    nens, npoints_ens = ens.shape
    assert npoints == npoints_ens
    frac = xp.zeros_like(clim)
    ##################################
    for icl in range(nclim):
        frac[icl, :] = xp.sum(ens[:, :] <= clim[icl, xp.newaxis, :], axis=0)
    ##################################
    frac /= nens

    # Compute formula coefficients
    p = xp.linspace(0.0, 1.0, nclim)
    dp = 1 / (nclim - 1)
    dFdp = xp.diff(frac, axis=0) / dp

    acosdiff = xp.diff(xp.arccos(xp.sqrt(p)))
    proddiff = xp.diff(xp.sqrt(p * (1.0 - p)))

    acoef = (1.0 - 2.0 * p[:-1]) * acosdiff + proddiff

    # compute EFI from coefficients
    efi = xp.zeros(npoints)
    ##################################
    if eps > 0:
        efimax = xp.zeros(npoints)
        for icl in range(nclim - 1):
            mask = clim[icl + 1, :] > eps
            dEFI = xp.where(
                mask,
                (2.0 * frac[icl, :] - 1.0) * acosdiff[icl] + acoef[icl] * dFdp[icl, :] - proddiff[icl],
                0.0,
            )
            defimax = xp.where(mask, -acosdiff[icl] - proddiff[icl], 0.0)
            efi += dEFI
            efimax += defimax
        efimax = xp.fmax(efimax, xp.asarray(eps))
        efi /= efimax
    else:
        for icl in range(nclim - 1):
            dEFI = (2.0 * frac[icl, :] - 1.0) * acosdiff[icl] + acoef[icl] * dFdp[icl, :] - proddiff[icl]
            efi += dEFI
        efi *= 2.0 / xp.pi
    ##################################

    # apply missing values
    efi[missing_mask] = xp.nan

    return efi


# @numba.jit(float64[:](float64[:,:], float64[:,:]), fastmath=False, nopython=True, nogil=True, cache=True)
# @numba.jit(nopython=True)
# def efi_numba(clim, ens):
#     """Compute EFI

#     Parameters
#     ----------
#     clim: numpy array (nclim, npoints)
#         Sorted per-point climatology
#     ens: numpy array (nens, npoints)
#         Ensemble forecast

#     Returns
#     -------
#     numpy array (npoints)
#         EFI values
#     """

#     # Compute fraction of the forecast below climatology
#     nclim, npoints = clim.shape
#     nens, npoints_ens = ens.shape
#     assert npoints == npoints_ens
#     frac = np.zeros_like(clim)
#     ##################################
#     for ifo in numba.prange(nens):
#         for icl in range(nclim):
#             for i in range(npoints):
#                if ens[ifo, i] <= clim[icl, i]:
#                    frac[icl, i] += 1
#     ##################################
#     frac /= nens

#     # Compute formula coefficients
#     p = np.linspace(0., 1., nclim)
#     dp = 1 / (nclim - 1)  #np.diff(p)

#     acosdiff = np.diff(np.arccos(np.sqrt(p)))
#     proddiff = np.diff(np.sqrt(p * (1. - p)))

#     acoef = (1. - 2. * p[:-1]) * acosdiff + proddiff

#     # TODO: handle epsilon
#     efi = np.zeros(npoints)
#     ##################################
#     for icl in numba.prange(nclim-1):
#         for i in range(npoints):
#             dFdp = (frac[icl+1, i] - frac[icl, i]) / dp
#              # XXX: why proddiff here?!
#             dEFI = (2. * frac[icl, i] - 1.) * acosdiff[icl] + acoef[icl] * dFdp - proddiff[icl]
#             efi[i] += dEFI
#     efi *= 2. / np.pi
#     ##################################

#     return efi
