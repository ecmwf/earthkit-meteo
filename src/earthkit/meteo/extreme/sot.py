# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import TYPE_CHECKING, overload

from earthkit.meteo.utils.decorators import dispatch

from . import array  # noqa

if TYPE_CHECKING:
    import xarray  # type: ignore[import]


@overload
def sot(
    clim: "xarray.DataArray",
    ens: "xarray.DataArray",
    perc: int,
    eps: float = -1e4,
    clim_dim: str | None = None,
    ens_dim: str | None = None,
) -> "xarray.DataArray": ...


def sot(
    clim,
    ens,
    perc: int,
    eps: float = -1e4,
    clim_dim: str | None = None,
    ens_dim: str | None = None,
):
    r"""Compute Shift of Tails (SOT)
    from climatology percentiles (sorted)
    and ensemble forecast (not sorted)

    Parameters
    ----------
    clim: xarray.DataArray
        Model climatology (percentiles). The reduction dimension is set by ``clim_dim``.
    ens: xarray.DataArrays
        Ensemble forecast. The reduction dimension is set by ``ens_dim``.
    perc: int
        Percentile value (typically 10 or 90)
    eps: (float)
        Epsilon factor for zero values
    clim_dim: str, optional
        Name of the climatology/quantile dimension in ``clim``.
    ens_dim: str, optional
        Name of the ensemble/member dimension in ``ens``.

    Returns
    -------
    xarray.DataArray
        SOT values.


    Implementations
    ------------------------
    :func:`sot` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.extreme.xarray.sot` for xarray.DataArray
    - :py:meth:`earthkit.meteo.extreme.array.sot` for array-like
    """
    res = dispatch(sot, clim, ens, perc, eps=eps, clim_dim=clim_dim, ens_dim=ens_dim)
    return res


@overload
def sot_unsorted(
    clim: "xarray.DataArray",
    ens: "xarray.DataArray",
    perc: int,
    eps: float = -1e4,
    clim_dim: str | None = None,
    ens_dim: str | None = None,
) -> "xarray.DataArray": ...


def sot_unsorted(
    clim,
    ens,
    perc: int,
    eps: float = -1e4,
    clim_dim: str | None = None,
    ens_dim: str | None = None,
):
    r"""Compute Shift of Tails (SOT)
    from climatology percentiles (sorted)
    and ensemble forecast (not sorted)

    Parameters
    ----------
    clim: xarray.DataArray
        Model climatology (percentiles). The reduction dimension is set by ``clim_dim``.
    ens: xarray.DataArray
        Ensemble forecast. The reduction dimension is set by ``ens_dim``.
    perc: int
        Percentile value (typically 10 or 90)
    eps: (float)
        Epsilon factor for zero values
    clim_dim: str, optional
        Name of the climatology/quantile dimension in ``clim``.
    ens_dim: str, optional
        Name of the ensemble/member dimension in ``ens``.

    Returns
    -------
    xarray.DataArray
        SOT values.


    Implementations
    ------------------------
    :func:`sot_unsorted` calls one of the following implementations depending on
    the type of the input arguments:

    - :py:meth:`earthkit.meteo.extreme.xarray.sot_unsorted` for xarray.DataArray
    - :py:meth:`earthkit.meteo.extreme.array.sot_unsorted` for array-like
    """
    res = dispatch(
        sot_unsorted, clim, ens, perc, eps=eps, clim_dim=clim_dim, ens_dim=ens_dim
    )
    return res
