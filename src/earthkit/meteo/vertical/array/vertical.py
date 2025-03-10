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

from earthkit.meteo import constants
from earthkit.meteo.thermo import specific_gas_consant

def pressure_at_model_levels(
    A: NDArray[Any], B: NDArray[Any], sp: Union[float, NDArray[Any]]
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
    sp : number or ndarray
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
    ndim = sp.ndim
    new_shape_half = (A.shape[0],) + (1,) * ndim
    A_reshaped = A.reshape(new_shape_half)
    B_reshaped = B.reshape(new_shape_half)

    # calculate pressure on model half-levels
    p_half_level = A_reshaped + B_reshaped * sp[np.newaxis, ...]

    # calculate delta
    new_shape_full = (A.shape[0] - 1,) + sp.shape
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

def relative_geopotential_thickness(alpha: NDArray[Any], q: NDArray[Any], T: NDArray[Any]) -> NDArray[Any]:
    """Calculates the geopotential thickness w.r.t the surface on model full-levels.

    Parameters
    ----------
    alpha : ndarray
        alpha term of pressure calculations
    q : ndarray
        specific humidity (in kg/kg) on model full-levels
    T : ndarray
        temperature (in Kelvin) on model full-levels

    Returns
    -------
    ndarray
        geopotential thickness of model full-levels w.r.t. the surface
    """

    R = calc_specific_gas_constant(q)
    dphi = np.cumsum(np.flip(alpha * R * T, axis=0), axis=0)
    dphi = np.flip(dphi, axis=0)

    return dphi

def pressure_at_height_level(
    height: float, q: NDArray[Any], T: NDArray[Any], sp: NDArray[Any], A: NDArray[Any], B: NDArray[Any]
) -> Union[float, NDArray[Any]]:
    """Calculates the pressure at a height level given in meters above surface.
    This is done by finding the model level above and below the specified height
    and interpolating the pressure.

    Parameters
    ----------
    height : number
        height (in meters) above the surface for which the pressure needs to be computed
    q : ndarray
        specific humidity (kg/kg) at model full-levels
    T : ndarray
        temperature (K) at model full-levels
    sp : ndarray
        surface pressure (Pa)
    A : ndarray
        A-coefficients defining the model levels
    B : ndarray
        B-coefficients defining the model levels

    Returns
    -------
    number or ndarray
        pressure (Pa) at the given height level
    """

    # geopotential thickness of the height level
    tdphi = height * constants.g

    # pressure(-related) variables
    p_full, p_half, _, alpha = model_level_pressure(A, B, sp)

    # relative geopot. thickness of full levels
    dphi = relative_geopotential_thickness(alpha, q, T)

    # find the model full level right above the height level
    i_phi = (tdphi > dphi).sum(0)

    # initialize the output array
    p_height = np.zeros_like(i_phi, dtype=np.float64)

    # define mask: requested height is below the lowest model full-level
    mask = i_phi == 0

    # CASE 1: requested height is below the lowest model full-level
    # --> interpolation between surface pressure and lowest model full-level
    p_height[mask] = (p_half[-1, ...] + tdphi / dphi[-1, ...] * (p_full[-1, ...] - p_half[-1, ...]))[mask]

    # CASE 2: requested height is above the lowest model full-level
    # --> interpolation between between model full-level above and below

    # define some indices for masking and readability
    i_lev = alpha.shape[0] - i_phi - 1  # convert phi index to model level index
    indices = np.indices(i_lev.shape)
    masked_indices = tuple(dim[~mask] for dim in indices)
    above = (i_lev[~mask],) + masked_indices
    below = (i_lev[~mask] + 1,) + masked_indices

    dphi_above = dphi[above]
    dphi_below = dphi[below]

    factor = (tdphi - dphi_above) / (dphi_below - dphi_above)
    p_height[~mask] = p_full[above] + factor * (p_full[below] - p_full[above])

    return p_height
