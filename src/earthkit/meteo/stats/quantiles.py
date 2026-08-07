# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Iterable, Sequence, TypeAlias, overload

from ..utils.decorators import dispatch

ArrayLike: TypeAlias = Any


@overload
def iter_quantiles(
    arr: "ArrayLike",
    which: int = 100,
    dim: int = 0,
    method: str = "sort",
) -> "Iterable[ArrayLike]": ...


@overload
def iter_quantiles(
    arr: "ArrayLike",
    which: Sequence[float],
    dim: int = 0,
    method: str = "sort",
) -> "Iterable[ArrayLike]": ...


def iter_quantiles(
    arr: "ArrayLike",
    which: int | Sequence[float] = 100,
    dim: int = 0,
    method: str = "sort",
) -> "Iterable[ArrayLike]":
    """Iterate over the quantiles of a large array.


    .. admonition:: Implementations

        Depending on the type of argument `arr`, this function calls:

        - :py:func:`earthkit.meteo.stats.array.iter_quantiles` for ``array_like``
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(iter_quantiles, array=True)
    return dispatched(arr, which=which, dim=dim, method=method)
