# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Iterable
from typing import List
from typing import Union

import numpy as np
from earthkit.utils.array import array_namespace


def iter_quantiles(
    arr: np.ndarray,
    which: Union[int, List[float]] = 100,
    axis: int = 0,
    method: str = "sort",
) -> Iterable[np.ndarray]:
    """Iterate over the quantiles of a large array

    Parameters
    ----------
    arr: numpy array
        Data array
    which: int or list of floats
        List of quantiles to compute, e.g. `[0., 0.25, 0.5, 0.75, 1.]`, or
        number of evenly-spaced intervals (e.g. 100 for percentiles).
    axis: int
        Axis along which to compute the quantiles
    method: 'sort', 'numpy_bulk', 'numpy'
        Method of computing the quantiles:
        * sort: sort `arr` in place, then interpolates the quantiles one by one
        * numpy_bulk: compute all the quantiles at once using `numpy.quantile`
        * numpy: compute the quantiles one by one using `numpy.quantile`

    Returns
    -------
    Iterable[numpy array]
        Quantiles, in increasing order if `which` is an `int`, otherwise in the order specified
    """

    if method not in ("sort", "numpy_bulk", "numpy"):
        raise ValueError(f"Invalid method {method!r}, expected 'sort', 'numpy_bulk', or 'numpy'")

    xp = array_namespace(arr)
    arr = xp.asarray(arr)

    if isinstance(which, int):
        n = which
        qs = xp.linspace(0.0, 1.0, n + 1)
    else:
        qs = xp.asarray(which)

    if method == "numpy_bulk":
        quantiles = xp.quantile(arr, qs, axis=axis)
        yield from quantiles
        return

    if method == "sort":
        arr = xp.asarray(arr)
        arr = xp.sort(arr, axis=axis)
        missing = xp.any(xp.isnan(arr), axis=axis)

    for q in qs:
        if method == "numpy":
            q = xp.asarray(q, dtype=arr.dtype)
            yield xp.quantile(arr, q, axis=axis)

        elif method == "sort":
            m = arr.shape[axis]
            f = (m - 1) * q
            j = int(f)
            x = f - j
            quantile = xp.take(arr, xp.asarray(j), axis=axis)
            quantile *= 1 - x
            tmp = xp.take(arr, xp.asarray(min(j + 1, m - 1)), axis=axis)
            tmp *= x
            quantile += tmp
            quantile[xp.reshape(missing, quantile.shape)] = xp.nan
            yield quantile
