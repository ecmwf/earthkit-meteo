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
def cpf(
    clim: "xarray.DataArray",
    ens: "xarray.DataArray",
    sort_clim: bool = True,
    sort_ens: bool = True,
    epsilon: float | None = None,
    symmetric: bool = False,
    from_zero: bool = False,
    clim_dim: str | None = None,
    ens_dim: str | None = None,
) -> "xarray.DataArray": ...


def cpf(
    clim,
    ens,
    sort_clim: bool = True,
    sort_ens: bool = True,
    epsilon: float | None = None,
    symmetric: bool = False,
    from_zero: bool = False,
    clim_dim: str | None = None,
    ens_dim: str | None = None,
):
    r"""Compute Crossing Point Forecast (CPF).

    WARNING: this code is experimental, use at your own risk!

    Parameters
    ----------
    clim: xarray.DataArray
        Per-point climatology. The reduction dimension (quantiles) is set by ``clim_dim``.
    ens: xarray.DataArray
        Ensemble forecast. The reduction dimension (ensemble members) is set by ``ens_dim``.
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
    clim_dim: str, optional
        Name of the climatology/quantile dimension in ``clim``.
    ens_dim: str, optional
        Name of the ensemble/member dimension in ``ens``.

    Returns
    -------
    xarray.DataArray
        CPF values.


    Implementations
    ------------------------
    :func:`cpf` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.extreme.xarray.cpf` for xarray.DataArray
    - :py:meth:`earthkit.meteo.extreme.array.cpf` for array-like
    """
    res = dispatch(
        cpf,
        clim,
        ens,
        sort_clim=sort_clim,
        sort_ens=sort_ens,
        epsilon=epsilon,
        symmetric=symmetric,
        from_zero=from_zero,
        clim_dim=clim_dim,
        ens_dim=ens_dim,
    )
    return res
