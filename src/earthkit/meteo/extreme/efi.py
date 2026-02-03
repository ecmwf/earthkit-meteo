# (C) Copyright 2021 ECMWF.
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
def efi(
    clim: "xarray.DataArray",
    ens: "xarray.DataArray",
    eps: float = -0.1,
    clim_dim: str | None = None,
    ens_dim: str | None = None,
) -> "xarray.DataArray": ...


def efi(
    clim,
    ens,
    eps: float = -0.1,
    clim_dim: str | None = None,
    ens_dim: str | None = None,
):
    r"""Compute Extreme Forecast Index (EFI).

    Parameters
    ----------
    clim: xarray.DataArray
        Sorted per-point climatology. The reduction dimension (quantiles) is set by ``clim_dim``.
    ens: xarray.DataArray
        Ensemble forecast. The reduction dimension (ensemble members) is set by ``ens_dim``.
    eps: (float)
        Epsilon factor for zero values
    clim_dim: str, optional
        Name of the climatology/quantile dimension in ``clim``.
    ens_dim: str, optional
        Name of the ensemble/member dimension in ``ens``.

    Returns
    -------
    xarray.DataArray
        EFI values.


    Implementations
    ------------------------
    :func:`efi` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.extreme.xarray.efi` for xarray.DataArray
    - :py:meth:`earthkit.meteo.extreme.array.efi` for array-like
    """
    res = dispatch(efi, clim, ens, eps=eps, clim_dim=clim_dim, ens_dim=ens_dim)
    return res
