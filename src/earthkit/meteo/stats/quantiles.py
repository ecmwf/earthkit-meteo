# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Any
from typing import Iterable
from typing import Sequence
from typing import TypeAlias
from typing import overload

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
    """Iterate over the quantiles of a large array


    .. admonition:: Implementations

        Depending on the type of argument `arr`, this function calls:

        - :py:func:`earthkit.meteo.stats.array.iter_quantiles` for ``array_like``
    The function returns an object of the same type as the input arguments.
    """
    dispatched = dispatch(iter_quantiles, array=True)
    return dispatched(arr, which=which, dim=dim, method=method)
