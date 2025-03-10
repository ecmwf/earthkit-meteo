# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.typing import NDArray

def pressure_at_model_levels(
    A: NDArray[Any], B: NDArray[Any], surface_pressure: Union[float, NDArray[Any]]
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    r"""Computes:
     - pressure at the model full- and half-levels
     - delta: depth of log(pressure) at full levels
     - alpha: alpha term #TODO: more descriptive information.

    Parameters
    ----------
    A : ndarray
        A-coefficients defining the model levels
    B : ndarray
        B-coefficients defining the model levels
    surface_pressure : number or ndarray
        surface pressure (Pa)

    Returns
    -------
    ndarray
        pressure at model full-levels
    ndarray
        pressure at model half-levels
    ndarray
        delta at full-levels
    ndarray
        alpha at full levels

    The pressure on the model-levels is calculated based on:

    .. math::

        p_{k+1/2} = A_{k+1/2} + p_{s} B_{k+1/2}
        p_k = 0.5 (p_{k-1/2} + p_{k+1/2})
    """

    # constants
    PRESSURE_TOA = 0.1  # safety when highest pressure level = 0.0

    # make the calculation agnostic to the number of dimensions
    ndim = surface_pressure.ndim
    new_shape_half = (A.shape[0],) + (1,) * ndim
    A_reshaped = A.reshape(new_shape_half)
    B_reshaped = B.reshape(new_shape_half)

    # calculate pressure on model half-levels
    p_half_level = A_reshaped + B_reshaped * surface_pressure[np.newaxis, ...]

    # calculate delta
    new_shape_full = (A.shape[0] - 1,) + surface_pressure.shape
    delta = np.zeros(new_shape_full)
    delta[1:, ...] = np.log(p_half_level[2:, ...] / p_half_level[1:-1, ...])

    # pressure at highest half level<= 0.1
    if np.any(p_half_level[0, ...] <= PRESSURE_TOA):
        delta[0, ...] = np.log(p_half_level[1, ...] / PRESSURE_TOA)
    # pressure at highest half level > 0.1
    else:
        delta[0, ...] = np.log(p_half_level[1, ...] / p_half_level[0, ...])

    # calculate alpha
    alpha = np.zeros(new_shape_full)

    alpha[1:, ...] = 1.0 - p_half_level[1:-1, ...] / (p_half_level[2:, ...] - p_half_level[1:-1, ...]) * delta[1:, ...]

    # pressure at highest half level <= 0.1
    if np.any(p_half_level[0, ...] <= PRESSURE_TOA):
        alpha[0, ...] = 1.0  # ARPEGE choice, ECMWF IFS uses log(2)
    # pressure at highest half level > 0.1
    else:
        alpha[0, ...] = 1.0 - p_half_level[0, ...] / (p_half_level[1, ...] - p_half_level[0, ...]) * delta[0, ...]

    # calculate pressure on model full levels
    # TODO: is there a faster way to calculate the averages?
    # TODO: introduce option to calculate full levels in more complicated way
    p_full_level = np.apply_along_axis(lambda m: np.convolve(m, np.ones(2) / 2, mode="valid"), axis=0, arr=p_half_level)

    return p_full_level, p_half_level, delta, alpha