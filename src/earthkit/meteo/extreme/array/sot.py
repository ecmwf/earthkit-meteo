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


def sot_func(qc_tail, qc, qf, eps=-1e-4, lower_bound=-10, upper_bound=10):
    """Compute basic Shift of Tails (SOT) using already computed percentiles

    Parameters
    ----------
    qc_tail: array-like (npoints)
        Tail percentile value (99% or 1%)
        Model climatology
    qc: array-like (npoints)
        Upper or lower percentile (at 90% or 10%)
        Model climatology
    qf: array-like(npoints)
        Upper or lower percentile (at 90% or 10%)
        Ensemble forecast
    eps: (float)
        Epsilon factor for zero values
    missing: (float)
        missing points values where denominator is zero

    Returns
    -------
    array-like (npoints)
        SOT values
    """
    xp = array_namespace(qc_tail, qc, qf)
    qc_tail = xp.asarray(qc_tail)
    qc = xp.asarray(qc)
    qf = xp.asarray(qf)

    min_den = max(eps, 0)
    sot = xp.where(xp.abs(qc_tail - qc) > min_den, (qf - qc_tail) / (qc_tail - qc), xp.nan)

    sot[sot < lower_bound] = lower_bound
    sot[sot > upper_bound] = upper_bound

    return sot


def sot(clim, ens, perc, eps=-1e4, clim_axis=0, ens_axis=0):
    """Compute Shift of Tails (SOT)
    from climatology percentiles (sorted)
    and ensemble forecast (not sorted)

    The reduction axis (ensemble and quantiles) is configurable by the user,
    but the other dimensions of clim and ens must be aligned and match.

    Parameters
    ----------
    clim: array-like
        Model climatology (percentiles). The reduction axis is set by ``clim_axis``.
    ens: array-like
        Ensemble forecast. The reduction axis is set by ``ens_axis``.
    perc: int
        Percentile value (typically 10 or 90)
    eps: (float)
        Epsilon factor for zero values
    clim_axis: int
        Axis index of the climatology/quantile dimension in ``clim``. Default is 0.
    ens_axis: int
        Axis index of the ensemble/member dimension in ``ens``. Default is 0.

    Returns
    -------
    array-like
        SOT values.
    """
    xp = array_namespace(clim, ens, perc)
    clim = xp.asarray(clim)
    ens = xp.asarray(ens)
    validate_extreme_shapes(
        func="sot",
        clim_shape=clim.shape,
        ens_shape=ens.shape,
        clim_axis=clim_axis,
        ens_axis=ens_axis,
    )
    clim, out_shape = flatten_extreme_input(xp, clim, clim_axis)
    ens, _ = flatten_extreme_input(xp, ens, ens_axis)
    # perc = xp.asarray(perc)

    signed_int_dtypes = xp.__array_namespace_info__().dtypes(kind="signed integer")
    signed_int_dtypes = set(signed_int_dtypes.values())
    if xp.asarray(perc).dtype not in signed_int_dtypes or (perc < 2 or perc > 98):
        raise Exception(
            "Percentile value should be and Integer between 2 and 98, is {} {}, {}".format(
                xp.asarray(perc).dtype, perc, signed_int_dtypes
            )
        )

    if clim.shape[0] != 101:
        raise Exception(
            "Climatology array should contain 101 percentiles, it has {} values".format(clim.shape)
        )

    qc = clim[perc]
    # if eps>0, set to zero everything below eps
    if eps > 0:
        ens = xp.where(ens < eps, 0.0, ens)
        qc = xp.where(qc < eps, 0.0, qc)

    qf = xp.percentile(ens, q=perc, axis=0)
    if perc > 50:
        qc_tail = clim[99]
    elif perc < 50:
        qc_tail = clim[1]
    else:
        raise Exception(
            "Percentile value to be computed cannot be 50 for sot, has to be in the upper or lower half"
        )

    sot = sot_func(qc_tail, qc, qf, eps=eps)

    return xp.reshape(sot, out_shape)


def sot_unsorted(clim, ens, perc, eps=-1e4, clim_axis=0, ens_axis=0):
    """Compute Shift of Tails (SOT)
    from climatology percentiles (sorted)
    and ensemble forecast (not sorted)

    The reduction axis (ensemble and quantiles) is configurable by the user,
    but the other dimensions of clim and ens must be aligned and match.

    Parameters
    ----------
    clim: array-like
        Model climatology (percentiles). The reduction axis is set by ``clim_axis``.
    ens: array-like
        Ensemble forecast. The reduction axis is set by ``ens_axis``.
    perc: int
        Percentile value (typically 10 or 90)
    eps: (float)
        Epsilon factor for zero values
    clim_axis: int
        Axis index of the climatology/quantile dimension in ``clim``. Default is 0.
    ens_axis: int
        Axis index of the ensemble/member dimension in ``ens``. Default is 0.

    Returns
    -------
    array-like
        SOT values.
    """
    xp = array_namespace(clim, ens, perc)
    clim = xp.asarray(clim)
    ens = xp.asarray(ens)
    perc = xp.asarray(perc)
    validate_extreme_shapes(
        func="sot_unsorted",
        clim_shape=clim.shape,
        ens_shape=ens.shape,
        clim_axis=clim_axis,
        ens_axis=ens_axis,
    )
    clim, out_shape = flatten_extreme_input(xp, clim, clim_axis)
    ens, _ = flatten_extreme_input(xp, ens, ens_axis)

    signed_int_dtypes = xp.__array_namespace_info__().dtypes(kind="signed integer")
    signed_int_dtypes = set(signed_int_dtypes.values())
    print(signed_int_dtypes)
    if xp.asarray(perc).dtype not in signed_int_dtypes or (perc < 2 or perc > 98):
        raise Exception(
            "Percentile value should be and Integer between 2 and 98, is {} {}".format(
                xp.asarray(perc).dtype, perc
            )
        )

    if clim.shape[0] != 101:
        raise Exception(
            "Climatology array should contain 101 percentiles, it has {} values".format(clim.shape)
        )

    if eps > 0:
        ens = xp.where(ens < eps, 0.0, ens)
        clim = xp.where(clim < eps, 0.0, clim)

    qf = xp.percentile(ens, q=perc, axis=0)
    qc = xp.percentile(clim, q=perc, axis=0)
    if perc > 50:
        perc_tail = 99
    elif perc < 50:
        perc_tail = 1
    else:
        raise Exception(
            "Percentile value to be computed cannot be 50 for sot, has to be in the upper or lower half"
        )
    qc_tail = xp.percentile(clim, q=perc_tail, axis=0)

    sot = sot_func(qc_tail, qc, qf, eps=eps)

    return xp.reshape(sot, out_shape)
