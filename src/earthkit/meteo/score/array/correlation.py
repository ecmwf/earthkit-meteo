# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
##
import numpy as np


def pearson(x, y, axis=0):
    """Pearson correlation coefficient

    Parameters
    ----------
    x, y: numpy array
        Input arrays, shapes must match
    axis: int
        Axis along which to compute correlation

    Returns
    -------
    numpy array
        Correlation, shape is the same as x with ``axis`` removed
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape == y.shape
    x = x - np.mean(x, axis=axis, keepdims=True)
    y = y - np.mean(y, axis=axis, keepdims=True)
    x /= np.linalg.norm(x, axis=axis, keepdims=True)
    y /= np.linalg.norm(y, axis=axis, keepdims=True)
    x *= y
    return np.sum(x, axis=axis)
