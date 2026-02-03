# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.utils.array import array_namespace

from .utils import flatten_extreme_input
from .utils import validate_extreme_shapes


def _cpf(clim, ens, epsilon=None, from_zero=False):
    xp = array_namespace(clim, ens)
    clim = xp.asarray(clim)
    ens = xp.asarray(ens)
    device = xp.device(clim)

    nclim, npoints = clim.shape
    nens, _ = ens.shape

    cpf = xp.zeros(npoints, dtype=xp.float32, device=device)
    mask = xp.zeros(npoints, dtype=xp.bool, device=device)
    prim = xp.zeros(npoints, dtype=xp.bool, device=device)

    # start scanning ensemble from iq_start
    iq_start = 0 if from_zero else nens // 2

    for icl in range(1, nclim - 1):
        # quantile level of climatology
        tau_c = icl / (nclim - 1.0)
        for iq in range(iq_start, nens):
            # quantile level of forecast
            tau_f = (iq + 1.0) / (nens + 1.0)

            # quantile values of forecast and climatology
            qv_f = ens[iq, :]
            qv_c = clim[icl, :]

            # primary condition (to ensure crossing and not reverse-crossing)
            if tau_f < tau_c:
                idx = (qv_f >= qv_c) & (~mask)
                prim[idx] = True

            if tau_f >= tau_c:
                # lowest climate quantile: interpolate between 2 consecutive quantiles
                if iq < 2:
                    # quantile value and quantile level of climatology at previous
                    qv_c_2 = clim[icl - 1, :]
                    tau_c_2 = (icl - 1) / (nclim - 1)

                    # condition of crossing situtaion:
                    idx = (qv_f < qv_c) & (qv_c_2 < qv_c) & prim

                    # intersection between two lines
                    tau_i = (tau_c * (qv_c_2[idx] - qv_f[idx]) + tau_c_2 * (qv_f[idx] - qv_c[idx])) / (
                        qv_c_2[idx] - qv_c[idx]
                    )

                    # populate matrix, no values below 0
                    cpf[idx] = xp.maximum(tau_i, xp.asarray(0))
                    mask[idx] = True

                # check crossing cases
                idx = (qv_f < qv_c) & (~mask) & prim
                cpf[idx] = tau_f
                mask[idx] = True

                # largest climate quantile: interpolate
                if iq == nens - 1:
                    qv_c_2 = clim[nclim - 1, :]
                    tau_c_2 = 1.0

                    idx = (qv_f > qv_c) & (qv_c_2 > qv_c) & (~mask) & prim

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


def cpf(
    clim,
    ens,
    sort_clim=True,
    sort_ens=True,
    epsilon=None,
    symmetric=False,
    from_zero=False,
    clim_axis=0,
    ens_axis=0,
):
    """Compute Crossing Point Forecast (CPF).

    WARNING: this code is experimental, use at your own risk!

    Parameters
    ----------
    clim: array-like
        Per-point climatology. The reduction axis (quantiles) is set by ``clim_axis``.
    ens: array-like
        Ensemble forecast. The reduction axis (ensemble members) is set by ``ens_axis``.
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
    from_zero: bool
        If True, start looking for a crossing from the minimum, rather than the
        median
    clim_axis: int
        Axis index of the climatology/quantile dimension in ``clim``. Default is 0.
    ens_axis: int
        Axis index of the ensemble/member dimension in ``ens``. Default is 0.

    Returns
    -------
    array-like
        CPF values with the reduction axes removed.
    """
    xp = array_namespace(clim, ens)
    clim = xp.asarray(clim)
    ens = xp.asarray(ens)

    clim, out_shape = flatten_extreme_input(xp, clim, clim_axis)
    ens, ens_shape = flatten_extreme_input(xp, ens, ens_axis)
    validate_extreme_shapes(
        func="cpf",
        clim_shape=out_shape,
        ens_shape=ens_shape,
        clim_axis=clim_axis,
        ens_axis=ens_axis,
    )

    if sort_clim:
        clim = xp.sort(clim, axis=0)
    if sort_ens:
        ens = xp.sort(ens, axis=0)

    if symmetric:
        epsilon = None

    cpf_direct = _cpf(clim, ens, epsilon, from_zero)

    if symmetric:
        cpf_reverse = _cpf(-xp.flip(clim, axis=0), -xp.flip(ens, axis=0), from_zero=from_zero)
        mask = cpf_direct < 0.5
        cpf_direct[mask] = 1 - cpf_reverse[mask]

    return xp.reshape(cpf_direct, out_shape)
