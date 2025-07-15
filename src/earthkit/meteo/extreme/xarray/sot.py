# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import xarray as xr

from earthkit.utils.array import array_namespace

from earthkit.meteo.extreme.array.sot import sot_func

def sot(clim: xr.DataArray, ens: xr.DataArray, perc: int, eps: float = -1e4, ens_dim: str = "number", clim_dim: str = "quantile") -> xr.DataArray:
    """Compute Shift of Tails (SOT)
    from climatology percentiles (sorted)
    and ensemble forecast (not sorted)

    Parameters
    ----------
    clim: array-like (nclim, npoints)
        Model climatology (percentiles)
    ens: array-like (nens, npoints)
        Ensemble forecast
    perc: int
        Percentile value (typically 10 or 90)
    eps: (float)
        Epsilon factor for zero values

    Returns
    -------
    array-like (npoints)
        SOT values
    """
    xp = array_namespace(clim.values, ens.values, perc)

    if not (isinstance(perc, int) or isinstance(perc, xp.int64)) or (perc < 2 or perc > 98):
        raise Exception("Percentile value should be and Integer between 2 and 98, is {}".format(perc))

    if clim.shape[0] != 101:
        raise Exception(
            "Climatology array should contain 101 percentiles, it has {} values".format(clim.shape)
        )

    qc = clim[{clim_dim: perc}]
    # if eps>0, set to zero everything below eps
    if eps > 0:
        ens = xp.where(ens < eps, 0.0, ens)
        qc = xp.where(qc < eps, 0.0, qc)

    qf = ens.quantile(perc/100., dim=ens_dim)
    if perc > 50:
        qc_tail = clim[{clim_dim: 99}]
    elif perc < 50:
        qc_tail = clim[{clim_dim: 1}]
    else:
        raise Exception(
            "Percentile value to be computed cannot be 50 for sot, has to be in the upper or lower half"
        )

    sot = xr.zeros_like(ens[{ens_dim: 0}])
    sot[:] = sot_func(qc_tail.values, qc.values, qf.values, eps=eps)

    return sot


def sot_unsorted(clim: xr.DataArray, ens: xr.DataArray, perc: int, eps=-1e4, ens_dim: str = "number", clim_dim: str = "quantile") -> xr.DataArray:
    """Compute Shift of Tails (SOT)
    from climatology percentiles (sorted)
    and ensemble forecast (not sorted)

    Parameters
    ----------
    clim: array-like (nclim, npoints)
        Model climatology (percentiles)
    ens: array-like (nens, npoints)
        Ensemble forecast
    perc: int
        Percentile value (typically 10 or 90)
    eps: (float)
        Epsilon factor for zero values

    Returns
    -------
    array-like (npoints)
        SOT values
    """
    xp = array_namespace(clim.values, ens.values, perc)

    if not (isinstance(perc, int) or isinstance(perc, xp.int64)) or (perc < 2 or perc > 98):
        raise Exception("Percentile value should be and Integer between 2 and 98, is {}".format(perc))

    if clim.shape[0] != 101:
        raise Exception(
            "Climatology array should contain 101 percentiles, it has {} values".format(clim.shape)
        )

    if eps > 0:
        ens = xp.where(ens < eps, 0.0, ens)
        clim = xp.where(clim < eps, 0.0, clim)

    qf = ens.quantile(perc/100., dim=ens_dim)
    qc = clim.quantile(perc/100., dim=clim_dim)
    if perc > 50:
        perc_tail = 99
    elif perc < 50:
        perc_tail = 1
    else:
        raise Exception(
            "Percentile value to be computed cannot be 50 for sot, has to be in the upper or lower half"
        )
    qc_tail = clim.quantile(perc_tail/100., dim=clim_dim)

    sot = xr.zeros_like(ens[{ens_dim: 0}])
    sot[:] = sot_func(qc_tail, qc.values, qf.values, eps=eps)

    return sot
