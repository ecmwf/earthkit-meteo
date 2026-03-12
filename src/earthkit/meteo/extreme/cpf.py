# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import TYPE_CHECKING
from typing import Any
from typing import TypeAlias
from typing import overload

from earthkit.meteo.utils.decorators import dispatch

ArrayLike: TypeAlias = Any

if TYPE_CHECKING:
    import xarray  # type: ignore[import]


@overload
def cpf(
    clim: "ArrayLike",
    ens: "ArrayLike",
    sort_clim: bool = True,
    sort_ens: bool = True,
    epsilon: float | None = None,
    symmetric: bool = False,
    from_zero: bool = False,
    clim_dim: int | None = 0,
    ens_dim: int | None = 0,
) -> "ArrayLike": ...


@overload
def cpf(
    clim: "xarray.DataArray",
    ens: "xarray.DataArray",
    sort_clim: bool = True,
    sort_ens: bool = True,
    epsilon: float | None = None,
    symmetric: bool = False,
    from_zero: bool = False,
    clim_dim: str | int | None = None,
    ens_dim: str | int | None = None,
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
    clim_dim: str or int, optional
        Name (or dimension index for array-like) of the climatology/quantile dimension in ``clim``.
    ens_dim: str or int, optional
        Name (or dimension index for array-like) of the ensemble/member dimension in ``ens``.

    Returns
    -------
    xarray.DataArray
        CPF values.


    Implementations
    ------------------------
    :func:`cpf` calls one of the following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.extreme.xarray.cpf` for xarray.DataArray
    - :py:meth:`earthkit.meteo.extreme.array.cpf` for array-like

    The function returns an object of the same type as the input arguments.
    """
    return dispatch(cpf, array=True)(
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
