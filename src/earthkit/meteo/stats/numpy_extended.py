# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, overload

from ..utils.decorators import dispatch

ArrayLike: TypeAlias = Any

if TYPE_CHECKING:
    import xarray  # type: ignore[import]


@overload
def nanaverage(data: "ArrayLike", weights: "ArrayLike" | None = None, **kwargs) -> "ArrayLike": ...


@overload
def nanaverage(
    data: "xarray.DataArray",
    weights: "xarray.DataArray" | None = None,
    **kwargs,
) -> "xarray.DataArray": ...


@overload
def nanaverage(
    data: "xarray.Dataset",
    weights: "xarray.Dataset" | None = None,
    **kwargs,
) -> "xarray.Dataset": ...


def nanaverage(data, weights=None, **kwargs):
    """A merge of the functionality of np.nanmean and np.average.


    .. admonition:: Implementations

        Depending on the type of argument `data`, this function calls:

        - :py:func:`earthkit.meteo.stats.array.nanaverage` for ``array_like``
        - :py:func:`earthkit.meteo.stats.xarray.nanaverage` for ``xarray.DataArray`` and ``xarray.Dataset``
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(nanaverage, xarray=True, array=True)
    return dispatched(data, weights, **kwargs)
