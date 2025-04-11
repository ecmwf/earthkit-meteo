# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.utils.array import array_namespace


def _cpf(clim, ens, epsilon=None):
    xp = array_namespace(clim, ens)
    clim = xp.asarray(clim)
    ens = xp.asarray(ens)

    nclim, npoints = clim.shape
    nens, _ = ens.shape

    cpf = xp.ones(npoints, dtype=xp.float32)
    mask = xp.zeros(npoints, dtype=xp.bool)

    for icl in range(1, nclim - 1):
        # quantile level of climatology
        tau_c = icl / (nclim - 1.0)
        for iq in range(nens):
            # quantile level of forecast
            tau_f = (iq + 1.0) / (nens + 1.0)
            if tau_f >= tau_c:
                # quantile values of forecast and climatology
                qv_f = ens[iq, :]
                qv_c = clim[icl, :]

                # lowest climate quantile: interpolate between 2 consecutive quantiles
                if iq < 2:
                    # quantile value and quantile level of climatology at previous
                    qv_c_2 = clim[icl - 1, :]
                    tau_c_2 = (icl - 1) / (nclim - 1)

                    # condition of crossing situtaion:
                    idx = (qv_f < qv_c) & (qv_c_2 < qv_c)

                    # intersection between two lines
                    tau_i = (tau_c * (qv_c_2[idx] - qv_f[idx]) + tau_c_2 * (qv_f[idx] - qv_c[idx])) / (
                        qv_c_2[idx] - qv_c[idx]
                    )

                    # populate matrix, no values below 0
                    cpf[idx] = xp.maximum(tau_i, xp.asarray(0))
                    mask[idx] = True

                # check crossing cases
                idx = (qv_f < qv_c) & (~mask)
                cpf[idx] = tau_f
                mask[idx] = True

                # largest climate quantile: interpolate
                if iq == nens - 1:
                    qv_c_2 = clim[nclim - 1, :]
                    tau_c_2 = 1.0

                    idx = (qv_f > qv_c) & (qv_c_2 > qv_c) & (~mask)

                    tau_i = (tau_c * (qv_c_2[idx] - qv_f[idx]) + tau_c_2 * (qv_f[idx] - qv_c[idx])) / (
                        qv_c_2[idx] - qv_c[idx]
                    )

                    # populate matrix, no values above 1
                    cpf[idx] = xp.minimum(tau_i, xp.asarray(1))

                # speed up process
                break

    if epsilon is not None:
        # ens is assumed to be sorted at this point
        mask = ens[-1, :] < epsilon
        cpf[mask] = 0.0

    return cpf


def cpf(clim, ens, sort_clim=True, sort_ens=True, epsilon=None, symmetric=False):
    """Compute Crossing Point Forecast (CPF)

    WARNING: this code is experimental, use at your own risk!

    Parameters
    ----------
    clim: array-like (nclim, npoints)
        Per-point climatology
    ens: array-like (nens, npoints)
        Ensemble forecast
    sort_clim: bool
        If True, sort the climatology first
    sort_ens: bool
        If True, sort the ensemble first
    epsilon: float or None
        If set, use this as a threshold for low-signal regions. Ignored if
        `symmetric` is True
    symmetric: bool
        If True, make CPF values below 0.5 use a symmetric computation (CPF of
        opposite values)

    Returns
    -------
    array-like (npoints)
        CPF values
    """
    xp = array_namespace(clim, ens)
    clim = xp.asarray(clim)
    ens = xp.asarray(ens)

    _, npoints = clim.shape
    _, npoints_ens = ens.shape
    assert npoints == npoints_ens

    if sort_clim:
        clim = xp.sort(clim, axis=0)
    if sort_ens:
        ens = xp.sort(ens, axis=0)

    if symmetric:
        epsilon = None

    cpf_direct = _cpf(clim, ens, epsilon)

    if symmetric:
        cpf_reverse = _cpf(-clim[::-1, :], -ens[::-1, :])
        mask = cpf_direct < 0.5
        cpf_direct[mask] = 1 - cpf_reverse[mask]

    return cpf_direct
