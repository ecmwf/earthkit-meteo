# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

from earthkit.utils.array import array_namespace


def pearson_correlation(x, y, axis=0):
    """Compute pearson correlation coefficient.

    Parameters
    ----------
    x, y: array-like
        Input arrays, shapes must match
    axis: int
        Axis along which to compute correlation

    Returns
    -------
    array-like
        Correlation, shape is the same as x with ``axis`` removed
    """
    xp = array_namespace(x, y)
    x = xp.asarray(x)
    y = xp.asarray(y)
    assert x.shape == y.shape
    x = x - xp.mean(x, axis=axis, keepdims=True)
    y = y - xp.mean(y, axis=axis, keepdims=True)
    x /= xp.linalg.vector_norm(x, axis=axis, keepdims=True)
    y /= xp.linalg.vector_norm(y, axis=axis, keepdims=True)
    x *= y
    return xp.sum(x, axis=axis)
