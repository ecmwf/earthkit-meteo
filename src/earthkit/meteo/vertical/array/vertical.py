# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import namedtuple
from typing import Any
from typing import Tuple
from typing import Union

import deprecation
import numpy as np
from earthkit.utils.array import array_namespace
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from earthkit.meteo import constants

ScalarInfo = namedtuple("ScalarInfo", ["values", "source", "target"])


@deprecation.deprecated(deprecated_in="0.7", details="Use pressure_on_hybrid_levels instead.")
def pressure_at_model_levels(
    A: NDArray[Any], B: NDArray[Any], sp: Union[float, NDArray[Any]], alpha_top: str = "ifs"
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    r"""Compute pressure at model full- and half-levels.

    Parameters
    ----------
    A : ndarray
        A-coefficients defining the model levels. See [IFS-CY47R3-Dynamics]_
        (page 6) for details.
    B : ndarray
        B-coefficients defining the model levels. See [IFS-CY47R3-Dynamics]_
        (page 6) for details.
    sp : number or ndarray
        Surface pressure (Pa)
    alpha_top : str, optional
        Option to initialise alpha on the top of the model atmosphere (first half-level in vertical coordinate system). The possible values are:

        - "ifs": alpha is set to log(2). See [IFS-CY47R3-Dynamics]_ (page 7) for details.
        - "arpege": alpha is set to 1.0

    Returns
    -------
    ndarray
        Pressure at model full-levels
    ndarray
        Pressure at model half-levels
    ndarray
        Delta at full-levels
    ndarray
        Alpha at full levels


    Notes
    -----
    ``A`` and ``B`` must contain the same model half-levels in ascending order with
    respect to the model level number. The model level range must be contiguous and
    must include the bottom-most model half-level (surface), but not all the levels
    must be present. E.g. if the vertical coordinate system has 137 model levels using
    only a subset of levels between e.g. 137-96 is allowed.

    For details on the returned parameters see [IFS-CY47R3-Dynamics]_ (page 7-8).

    The pressure on the model-levels is calculated as:

    .. math::

        p_{k+1/2} = A_{k+1/2} + p_{s}\; B_{k+1/2}

        p_{k} = \frac{1}{2}\; (p_{k-1/2} + p_{k+1/2})

    where

        - :math:`p_{s}` is the surface pressure
        - :math:`p_{k+1/2}` is the pressure at the half-levels
        - :math:`p_{k}` is the pressure at the full-levels
        - :math:`A_{k+1/2}` and :math:`B_{k+1/2}` are the A- and B-coefficients defining
          the model levels.

    See also
    --------
    pressure_at_height_levels
    relative_geopotential_thickness

    """
    # constants
    PRESSURE_TOA = 0.1  # safety when highest pressure level = 0.0

    if alpha_top not in ["ifs", "arpege"]:
        raise ValueError(f"Unknown method '{alpha_top}' for pressure calculation. Use 'ifs' or 'arpege'.")

    alpha_top = np.log(2) if alpha_top == "ifs" else 1.0

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

    alpha[1:, ...] = (
        1.0 - p_half_level[1:-1, ...] / (p_half_level[2:, ...] - p_half_level[1:-1, ...]) * delta[1:, ...]
    )

    # pressure at highest half level <= 0.1
    if np.any(p_half_level[0, ...] <= PRESSURE_TOA):
        alpha[0, ...] = alpha_top
    # pressure at highest half level > 0.1
    else:
        alpha[0, ...] = (
            1.0 - p_half_level[0, ...] / (p_half_level[1, ...] - p_half_level[0, ...]) * delta[0, ...]
        )

    # calculate pressure on model full levels
    # TODO: is there a faster way to calculate the averages?
    # TODO: introduce option to calculate full levels in more complicated way
    p_full_level = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(2) / 2, mode="valid"), axis=0, arr=p_half_level
    )

    return p_full_level, p_half_level, delta, alpha


@deprecation.deprecated(
    deprecated_in="0.7", details="Use relative_geopotential_thickness_on_hybrid_levels instead."
)
def relative_geopotential_thickness(
    alpha: ArrayLike, delta: ArrayLike, t: ArrayLike, q: ArrayLike
) -> ArrayLike:
    """Calculate the geopotential thickness with respect to the surface on hybrid (IFS model) full-levels.

    Parameters
    ----------
    alpha : array-like
        Alpha term of pressure calculations
    delta : array-like
        Delta term of pressure calculations
    t : array-like
        Temperature on hybrid (IFS model) full-levels (K). First dimension must
        correspond to the full-levels.
    q : Specific humidity on hybrid (IFS model) full-levels (kg/kg). First dimension must
        correspond to the full-levels.

    Returns
    -------
    array-like
        Geopotential thickness (m2/s2) of hybrid (IFS model) full-levels with respect to the surface

    Notes
    -----
    ``t`` and ``q`` must contain the same levels in ascending order with respect to
    the level number. The model level range must be contiguous and must include the
    bottom-most level, but not all the levels must be present. E.g. if the vertical coordinate
    system has 137 model levels using only a subset of levels between e.g. 137-96 is allowed.

    ``alpha`` and ``delta`` must be defined on the same levels as ``t`` and ``q``. These
    values can be calculated using :func:`pressure_at_model_levels`.

    The computations are described in [IFS-CY47R3-Dynamics]_ (page 7-8).

    See also
    --------
    pressure_at_model_levels

    """
    from earthkit.meteo.thermo import specific_gas_constant

    xp = array_namespace(alpha, delta, q, t)

    R = specific_gas_constant(q)
    d = R * t

    # compute geopotential thickness on half levels from 1 to NLEV-1
    dphi_half = xp.cumulative_sum(xp.flip(d[1:, ...] * delta[1:, ...], axis=0), axis=0)
    dphi_half = xp.flip(dphi_half, axis=0)

    # compute geopotential thickness on full levels
    dphi = xp.zeros_like(d)
    dphi[:-1, ...] = dphi_half + d[:-1, ...] * alpha[:-1, ...]
    dphi[-1, ...] = d[-1, ...] * alpha[-1, ...]

    return dphi


def pressure_at_height_levels(
    height: float,
    t: NDArray[Any],
    q: NDArray[Any],
    sp: NDArray[Any],
    A: NDArray[Any],
    B: NDArray[Any],
    alpha_top: str = "ifs",
) -> Union[float, NDArray[Any]]:
    """Calculate the pressure at a height above the surface from model full-levels.

    Parameters
    ----------
    height : number
        height above the surface for which the pressure needs to be computed (m)
    t : ndarray
        temperature at model full-levels (K). First dimension must
        correspond to the model full-levels.
    q : ndarray
        specific humidity at model full-levels (kg/kg). First dimension must
        correspond to the model full-levels.
    sp : ndarray
        surface pressure (Pa)
    A : ndarray
        A-coefficients defining the model levels
    B : ndarray
        B-coefficients defining the model levels
    alpha_top : str, optional
        Option passed to :func:`pressure_at_model_levels`. The possible values
        are: "ifs" (default) or "arpege".

    Returns
    -------
    number or ndarray
        pressure at the given height level (Pa)

    Notes
    -----
    ``t`` and ``q`` must contain the same model levels in ascending order with respect to
    the model level number. The model level range must be contiguous and must include the
    bottom-most level, but not all the levels must be present. E.g. if the vertical coordinate
    system has 137 model levels using only a subset of levels between e.g. 137-96 is allowed.

    ``A`` and ``B`` must be defined on the model half-levels corresponding to the model
    full-levels in ``t`` and ``q``. So the number of levels in ``A`` and ``B`` must be one
    more than the number of levels in ``t`` and ``q``.

    The pressure at height level is calculated by finding the model level above and
    below the specified height and interpolating the pressure with linear interpolation.

    See also
    --------
    pressure_at_model_levels
    relative_geopotential_thickness


    """
    # geopotential thickness of the height level
    tdphi = height * constants.g

    nlev = A.shape[0] - 1  # number of model full-levels

    # pressure(-related) variables
    p_full, p_half, delta, alpha = pressure_at_model_levels(A, B, sp, alpha_top=alpha_top)

    # relative geopotential thickness of full levels
    dphi = relative_geopotential_thickness(alpha, delta, t, q)

    # find the model full level right above the height level
    i_phi = (tdphi < dphi).sum(0)
    i_phi = i_phi - 1

    # TODO: handle case when height is above the highest model full-level
    # TODO: handle case when height is below the surface

    # initialize the output array
    p_height = np.zeros_like(i_phi, dtype=np.float64)

    # define mask: requested height is below the lowest model full-level
    mask = i_phi == nlev - 1

    # CASE 1: requested height is below the lowest model full-level
    # --> interpolation between surface pressure and lowest model full-level
    p_height[mask] = (p_half[-1, ...] + tdphi / dphi[-1, ...] * (p_full[-1, ...] - p_half[-1, ...]))[mask]

    # CASE 2: requested height is above the lowest model full-level
    # --> interpolation between between model full-level above and below

    # define some indices for masking and readability
    i_lev = i_phi
    indices = np.indices(i_lev.shape)
    masked_indices = tuple(dim[~mask] for dim in indices)
    above = (i_lev[~mask],) + masked_indices
    below = (i_lev[~mask] + 1,) + masked_indices

    dphi_above = dphi[above]
    dphi_below = dphi[below]

    # print(
    #     f"tdphi: {tdphi} above: {above} below: {below} dphi_above: {dphi_above} dphi_below  {dphi_below} p_full[above]: {p_full[above]} p_full[below]: {p_full[below]}"
    # )

    # calculate the interpolation factor
    factor = (tdphi - dphi_above) / (dphi_below - dphi_above)
    p_height[~mask] = p_full[above] + factor * (p_full[below] - p_full[above])

    return p_height


def geopotential_height_from_geopotential(z):
    r"""Compute geopotential height from geopotential.

    Parameters
    ----------
    z : array-like
        Geopotential (m2/s2)

    Returns
    -------
    array-like
        Geopotential height (m)


    The computation is based on the following definition:

    .. math::

        gh = \frac{z}{g}

    where :math:`g` is the gravitational acceleration on the surface of
    the Earth (see :py:attr:`meteo.constants.g`)
    """
    h = z / constants.g
    return h


def geopotential_from_geopotential_height(h):
    r"""Compute geopotential height from geopotential.

    Parameters
    ----------
    z : array-like
        Geopotential (m2/s2)

    Returns
    -------
    array-like
        Geopotential height (m)


    The computation is based on the following definition:

    .. math::

        z = gh\; g

    where :math:`g` is the gravitational acceleration on the surface of
    the Earth (see :py:attr:`meteo.constants.g`)
    """
    z = h * constants.g
    return z


def geopotential_height_from_geometric_height(h, R_earth=constants.R_earth):
    r"""Compute the geopotential height from geometric height.

    Parameters
    ----------
    h : array-like
        Geometric height with respect to the sea level (m)
    R_earth : float, optional
        Average radius of the Earth (m)

    Returns
    -------
    array-like
        Geopotential height (m)


    The computation is based on the following formula:

    .. math::

        gh = \frac{h\; R_{earth}}{R_{earth} + h}

    where :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`meteo.constants.R_earth`)
    """
    zh = h * R_earth / (R_earth + h)
    return zh


def geopotential_from_geometric_height(h, R_earth=constants.R_earth):
    r"""Compute the geopotential from geometric height.

    Parameters
    ----------
    h : array-like
        Geometric height with respect to the sea level (m)
    R_earth : float, optional
        Average radius of the Earth (m)

    Returns
    -------
    array-like
        Geopotential (m2/s2)


    The computation is based on the following formula:

    .. math::

        z = \frac{h\; g\; R_{earth}}{R_{earth} + h}

    where

        * :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`meteo.constants.R_earth`)
        * :math:`g` is the gravitational acceleration on the surface of
          the Earth (see :py:attr:`meteo.constants.g`)
    """
    z = h * R_earth * constants.g / (R_earth + h)
    return z


def geometric_height_from_geopotential_height(gh, R_earth=constants.R_earth):
    r"""Compute the geometric height from geopotential height.

    Parameters
    ----------
    gh : array-like
        Geopotential height (m)
    R_earth : float, optional
        Average radius of the Earth (m)

    Returns
    -------
    array-like
        Geometric height (m)


    The computation is based on the following formula:

    .. math::

        h = \frac{R_{earth}\; gh}{R_{earth} - gh}

    where :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`meteo.constants.R_earth`)
    """
    h = R_earth * gh / (R_earth - gh)
    return h


def geometric_height_from_geopotential(z, R_earth=constants.R_earth):
    r"""Compute the geometric height from geopotential.

    Parameters
    ----------
    z : array-like
        Geopotential (m2/s2)
    R_earth : float, optional
        Average radius of the Earth (m)

    Returns
    -------
    array-like
        Geometric height (m)


    The computation is based on the following formula:

    .. math::

        h = \frac{R_{earth} \frac{z}{g}}{R_{earth} - \frac{z}{g}}

    where

        * :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`meteo.constants.R_earth`)
        * :math:`g` is the gravitational acceleration on the surface of
          the Earth (see :py:attr:`meteo.constants.g`)
    """
    z = z / constants.g
    h = R_earth * z / (R_earth - z)
    return h


def pressure_on_hybrid_levels(
    A: ArrayLike, B: ArrayLike, sp: ArrayLike, levels=None, alpha_top="ifs", output="full"
) -> ArrayLike:
    r"""Compute pressure and related parameters on hybrid (IFS model) levels.

    *New in version 0.7.0*: This function replaces the deprecated :func:`pressure_at_model_levels`.

    Parameters
    ----------
    A : array-like
        A-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number. If the total number
        of (full) model levels is :math:`NLEV`, ``A`` must contain :math:`NLEV+1` values, one for each
        half-level. See [IFS-CY47R3-Dynamics]_ (page 6) for details.
    B : array-like
        B-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number. Must have the same
        size as ``A``.  If the total number of (full) model levels is :math:`NLEV`, ``B`` must
        contain :math:`NLEV+1` values. See [IFS-CY47R3-Dynamics]_ (page 6) for details.
    sp : array-like
        Surface pressure (Pa)
    levels : None, array-like, list, tuple, optional
        Specify the full hybrid levels to return in the given order. Please note level
        numbering starts at 1. If None (default), all the levels are returned in the
        order defined by the A and B coefficients (i.e. ascending order with respect to
        the model level number).
    alpha_top : str, optional
        Option to initialise the alpha parameters (for details see below) on the top of the
        model atmosphere (first half-level in the vertical coordinate system). The possible
        values are:

        - "ifs": alpha is set to log(2). See [IFS-CY47R3-Dynamics]_ (page 7) for details.
        - "arpege": alpha is set to 1.0

    output : str or list/tuple of str, optional
        Specify which outputs to return. Possible values are "full", "half", "delta" and "alpha".
        Can be a single string or a list/tuple of strings. Default is "full". The outputs are:

        - "full": pressure (Pa) on full levels
        - "half": pressure (Pa) on half levels. When ``levels`` is None, returns all the
          half-levels. When ``levels`` is not None, only returns the half-levels below
          the requested full levels.
        - "delta": logarithm of pressure difference between two adjacent half-levels. Uses
          the same indexing as
          the full levels.
        - "alpha": alpha parameter defined for layers (i.e. for full levels). Uses the same
          indexing as the full levels. Used for the calculation of the relative geopotential
          thickness on full levels. See :func:`relative_geopotential_thickness` for details.


    Returns
    -------
    array-like or tuple of array-like
        See the ``output`` parameter for details.

    Notes
    -----
    The hybrid model levels divide the atmosphere into :math:`NLEV` layers. These layers are defined
    by the pressures at the interfaces between them for :math:`0 \leq k \leq NLEV`, which are
    the half-levels :math:`p_{k+1/2}` (indices increase from the top of the atmosphere towards
    the surface). The half levels are defined by the ``A`` and ``B`` coefficients in such a way that at
    the top of the atmosphere the first half level pressure :math:`p_{+1/2}` is a constant, while
    at the surface :math:`p_{NLEV+1/2}` is the surface pressure.

    The full-level pressure :math:`p_{k}` associated with each model
    level is defined as the middle of the layer for :math:`1 \leq k \leq NLEV`.

    The level definitions can be written as:

    .. math::

        p_{k+1/2} = A_{k+1/2} + p_{s}\; B_{k+1/2}

        p_{k} = \frac{1}{2}\; (p_{k-1/2} + p_{k+1/2})

    where

        - :math:`p_{s}` is the surface pressure
        - :math:`p_{k+1/2}` is the pressure at the half-levels
        - :math:`p_{k}` is the pressure at the full-levels
        - :math:`A_{k+1/2}` and :math:`B_{k+1/2}` are the A- and B-coefficients defining
          the model levels.

    For more details see [IFS-CY47R3-Dynamics]_ (page 6-8).

    See also
    --------
    pressure_at_height_levels
    relative_geopotential_thickness

    """
    if isinstance(output, str):
        output = (output,)

    if not output:
        raise ValueError("At least one output type must be specified.")

    for out in output:
        if out not in ["full", "half", "alpha", "delta"]:
            raise ValueError(
                f"Unknown output type '{out}'. Allowed values are 'full', 'half', 'alpha' or 'delta'."
            )

    if alpha_top not in ["ifs", "arpege"]:
        raise ValueError(f"Unknown method '{alpha_top}' for pressure calculation. Use 'ifs' or 'arpege'.")

    if levels is not None:
        # select a contiguous subset of levels
        nlev = A.shape[0] - 1  # number of model full-levels
        levels = np.asarray(levels)
        levels_max = int(levels.max())
        levels_min = int(levels.min())
        if levels_max > nlev:
            raise ValueError(f"Requested level {levels_max} exceeds the maximum number of levels {nlev}.")
        if levels_min < 1:
            raise ValueError(f"Level numbering starts at 1. Found level={levels_min} < 1.")

        half_idx = np.array(list(range(levels_min - 1, levels_max + 1)))
        A = A[half_idx]
        B = B[half_idx]

        # compute indices to select the requested full levels later
        out_half_idx = np.where(levels[:, None] == half_idx[None, :])[1]
        out_full_idx = out_half_idx - 1

        print(f"Selected levels: {levels} half_idx: {half_idx} out_full_idx: {out_full_idx}")

    # make the calculation agnostic to the number of dimensions
    ndim = sp.ndim
    new_shape_half = (A.shape[0],) + (1,) * ndim
    A_reshaped = A.reshape(new_shape_half)
    B_reshaped = B.reshape(new_shape_half)

    # calculate pressure on model half-levels
    p_half_level = A_reshaped + B_reshaped * sp[np.newaxis, ...]

    if "delta" in output or "alpha" in output:
        # constants
        PRESSURE_TOA = 0.1  # safety when highest pressure level = 0.0

        alpha_top = np.log(2) if alpha_top == "ifs" else 1.0

        new_shape_full = (A.shape[0] - 1,) + sp.shape

        # calculate delta
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

        alpha[1:, ...] = (
            1.0 - p_half_level[1:-1, ...] / (p_half_level[2:, ...] - p_half_level[1:-1, ...]) * delta[1:, ...]
        )

        # pressure at highest half level <= 0.1
        if np.any(p_half_level[0, ...] <= PRESSURE_TOA):
            alpha[0, ...] = alpha_top
        # pressure at highest half level > 0.1
        else:
            alpha[0, ...] = (
                1.0 - p_half_level[0, ...] / (p_half_level[1, ...] - p_half_level[0, ...]) * delta[0, ...]
            )

    if "full" in output:
        # calculate pressure on model full levels
        # TODO: is there a faster way to calculate the averages?
        # TODO: introduce option to calculate full levels in more complicated way
        p_full_level = np.apply_along_axis(
            lambda m: np.convolve(m, np.ones(2) / 2, mode="valid"), axis=0, arr=p_half_level
        )

    # generate output
    res = []

    for out in output:
        if out == "full":
            if levels is not None:
                p_full_level = p_full_level[out_full_idx, ...]
            res.append(p_full_level)
        elif out == "half":
            if levels is not None:
                p_half_level = p_half_level[out_half_idx, ...]
            res.append(p_half_level)
        elif out == "alpha":
            if levels is not None:
                alpha = alpha[out_full_idx, ...]
            res.append(alpha)
        elif out == "delta":
            if levels is not None:
                delta = delta[out_full_idx, ...]
            res.append(delta)

    if len(res) == 1:
        return res[0]

    return tuple(res)


def relative_geopotential_thickness_on_hybrid_levels(
    t: ArrayLike,
    q: ArrayLike,
    alpha: ArrayLike,
    delta: ArrayLike,
) -> ArrayLike:
    """Calculate the geopotential thickness with respect to the surface on hybrid (IFS model) full-levels.

    Parameters
    ----------
    t : array-like
        Temperature on hybrid (IFS model) full-levels (K). First dimension must
        correspond to the full-levels.
    q : Specific humidity on hybrid (IFS model) full-levels (kg/kg). First dimension must
        correspond to the full-levels.
    alpha : array-like
        Alpha term of pressure calculations computed using :func:`pressure_on_hybrid_levels`
    delta : array-like
        Delta term of pressure calculations computed using :func:`pressure_on_hybrid_levels`

    Returns
    -------
    array-like
        Geopotential thickness (m2/s2) of hybrid (IFS model) full-levels with respect to the surface

    Notes
    -----
    ``t`` and ``q`` must contain the same model levels in ascending order with respect to
    the model level number. The model level range must be contiguous and must include the
    bottom-most level, but not all the levels must be present. E.g. if the vertical coordinate
    system has 137 model levels using only a subset of levels between e.g. 137-96 is allowed.

    ``alpha`` and ``delta`` must be defined on the same levels as ``t`` and ``q``. These
    values can be calculated using :func:`pressure_on_hybrid_levels`.

    The computations are described in [IFS-CY47R3-Dynamics]_ (page 7-8).

    See also
    --------
    pressure_on_hybrid_levels

    """
    from earthkit.meteo.thermo import specific_gas_constant

    xp = array_namespace(alpha, delta, q, t)

    R = specific_gas_constant(q)
    d = R * t

    # compute geopotential thickness on half levels from 1 to NLEV-1
    dphi_half = xp.cumulative_sum(xp.flip(d[1:, ...] * delta[1:, ...], axis=0), axis=0)
    dphi_half = xp.flip(dphi_half, axis=0)

    # compute geopotential thickness on full levels
    dphi = xp.zeros_like(d)
    dphi[:-1, ...] = dphi_half + d[:-1, ...] * alpha[:-1, ...]
    dphi[-1, ...] = d[-1, ...] * alpha[-1, ...]

    return dphi


def geopotential_on_hybrid_levels(
    t: ArrayLike,
    q: ArrayLike,
    zs: ArrayLike,
    alpha: ArrayLike,
    delta: ArrayLike,
):

    xp = array_namespace(t, q, zs, alpha, delta)
    t = xp.asarray(t)
    q = xp.asarray(q)
    zs = xp.asarray(zs)
    alpha = xp.asarray(alpha)
    delta = xp.asarray(delta)

    phi = relative_geopotential_thickness_on_hybrid_levels(t, q, alpha, delta)
    return phi + zs


def interpolate_monotonic(
    data: ArrayLike,
    coord_data: Union[ArrayLike, list, tuple, float, int],
    coord_target: Union[ArrayLike, list, tuple, float, int],
    interpolation: str = "linear",
) -> ArrayLike:
    """Interpolate data onto monotonic coordinate levels.

    Parameters
    ----------
    data : array-like
        Data to be interpolated. First dimension must correspond to the vertical. Must have at
        least two levels. Levels must be ordered in ascending or descending order.
    coord_data : array-like
        Monotonic coordinate levels corresponding to the first dimension of ``data``. Either must
        have the same shape as ``data`` or be a 1D array with length equal to the size of the first
        dimension of ``data``.
    coord_target : array-like
        Target coordinate levels to which ``data`` will be interpolated. It can be either a scalar
        or a 1D array of coordinate levels. Alternatively, it can be an array of arrays where each sub-array
        contains the target coordinate levels for the corresponding horizontal location in ``data``. The
        units are in the same units as ``coord_data``.
    interpolation  : str, optional
        Interpolation mode. Default is "linear". Possible values are:
        - "linear": linear interpolation in coordinate
        - "log": linear interpolation in logarithm of coordinate
        - "nearest": nearest neighbour interpolation

    Returns
    -------
    array-like
        Data interpolated to the target levels. The shape depends on the shape of ``coord_target``:
        - If ``coord_target`` is a scalar, the output shape is equal to ``data.shape[1:]``.
        - If ``coord_target`` is a 1D array of length N, the output shape is (N, ) + ``data.shape[1:]``.
        - If ``coord_target`` is an array of arrays, the output shape is (M, ) + ``data.shape[1:]``,
          where M is the number of horizontal locations in ``data``.
        When interpolation is not possible for a given target coordinate level (e.g., when the target coordinate
        is outside the range of ``coord_data``), the corresponding output values are set to NaN.

    Raises
    ------
    ValueError
        If ``data`` has less than two levels.
    ValueError
        If the first dimension of ``data`` and that of ``coord_data`` do not match.


    Notes
    -----
    - The ordering of the input coordinate levels is not checked.
    - The units of ``coord_data`` and ``coord_target`` are assumed to be the same; no checks
      or conversions are performed.

    """

    xp = array_namespace(data, coord_data)
    coord_target = xp.atleast_1d(coord_target)
    coord_data = xp.atleast_1d(coord_data)

    # Ensure levels are in descending order with respect to the first dimension
    first = [0] * xp.ndim(coord_data)
    last = [0] * xp.ndim(coord_data)
    first = tuple([0] + first[1:])
    last = tuple([-1] + last[1:])

    if coord_data[first] < coord_data[last]:
        coord_data = xp.flip(coord_data, axis=0)
        data = xp.flip(data, axis=0)

    nlev = data.shape[0]
    if nlev < 2:
        raise ValueError("At least two levels are required for interpolation.")

    if data.shape[0] != coord_data.shape[0]:
        raise ValueError(
            f"The first dimension of data and that of coord_data must match! {data.shape=} {coord_data.shape=} {data.shape[0]} != {coord_data.shape[0]}"
        )

    scalar_info = ScalarInfo(
        xp.ndim(data[0]) == 0, xp.ndim(coord_data[0]) == 0, xp.ndim(coord_target[0]) == 0
    )

    data_same_shape = data.shape == coord_data.shape
    if data_same_shape:
        if scalar_info.values and not scalar_info.target:
            raise ValueError("If values and p have the same shape, they cannot both be scalars.")
        if not scalar_info.values and not scalar_info.target and data.shape[1:] != coord_target.shape[1:]:
            raise ValueError(
                "When values and target_p have different shapes, target_p must be a scalar or a 1D array."
            )

    if not data_same_shape and xp.ndim(coord_data) != 1:
        raise ValueError(
            f"When values and p have different shapes, p must be a scalar or a 1D array. {data.shape=} {coord_data.shape=} {xp.ndim(coord_data)}"
        )

    # initialize the output array
    res = xp.empty((len(coord_target),) + data.shape[1:], dtype=data.dtype)
    if data_same_shape:
        if scalar_info.values:
            data = xp.broadcast_to(data, (1, nlev)).T
            coord_data = xp.broadcast_to(coord_data, (1, nlev)).T
        else:
            assert not scalar_info.values
            assert not scalar_info.source
    else:
        assert scalar_info.source
        # print(f"scalar_info.target: {scalar_info.target}")
        if scalar_info.target:
            return _to_level_1(data, coord_data, nlev, coord_target, interpolation, scalar_info, xp, res)
        else:
            coord_data = xp.broadcast_to(coord_data, (nlev,) + data.shape[1:]).T

    return _to_level(data, coord_data, nlev, coord_target, interpolation, scalar_info, xp, res)


# values and p have the same shape
def _to_level(data, src_coord, nlev, coord_target, interpolation, scalar_info, xp, res):
    # The coordinate levels must be ordered in descending order with respect to the
    # first dimension. So index 0 has the highest coordinate values, index -1 the lowest,
    # as if it were pressure levels in the atmosphere. The algorithm below agnostic to the
    # actual meaning of the coordinate in the real atmosphere. The terms "top" and "bottom"
    # are used with respect to this coordinate ordering in mind and not related to actual
    # vertical position in the atmosphere. Of course, if the  coordinate is pressure these
    # two definitions coincide.
    for target_idx, tc in enumerate(coord_target):

        # find the level below the target
        idx_bottom = (src_coord > tc).sum(0)
        idx_bottom = xp.atleast_1d(idx_bottom)

        # print(f"tc: {tc} i_top: {i_top}")
        # initialise the output array
        r = xp.empty(idx_bottom.shape)

        # mask when the target is below the lowest level
        mask_bottom = idx_bottom == 0
        if xp.any(mask_bottom):
            if interpolation != "nearest":
                r[mask_bottom] = np.nan
                m = mask_bottom & (src_coord[0] == tc)
                r[m] = data[0][m]

        # mask when the target is above the highest level
        mask_top = idx_bottom == nlev
        if xp.any(mask_top):
            if interpolation != "nearest":
                r[mask_top] = np.nan
                m = mask_top & (src_coord[-1] == tc)
                r[m] = data[-1][m]

        # mask when the target is in the coordinate range
        mask_mid = ~(mask_bottom | mask_top)

        if xp.any(mask_mid):
            i_lev = idx_bottom
            indices = np.indices(i_lev.shape)
            masked_indices = tuple(dim[mask_mid] for dim in indices)
            top = (idx_bottom[mask_mid],) + masked_indices
            bottom = (idx_bottom[mask_mid] - 1,) + masked_indices
            c_top = src_coord[top]
            c_bottom = src_coord[bottom]

            f_top = data[top]
            f_bottom = data[bottom]

            # print(f"tc: {tc} c_top: {c_top} c_bottom: {c_bottom} f_top: {f_top} f_bottom: {f_bottom}")

            if not scalar_info.target:
                tc = tc[mask_mid]

            # calculate the interpolation factor
            if interpolation == "linear":
                factor = (tc - c_bottom) / (c_top - c_bottom)
            elif interpolation == "log":
                factor = (xp.log(tc) - xp.log(c_bottom)) / (xp.log(c_top) - xp.log(c_bottom))

            r[mask_mid] = (1.0 - factor) * f_bottom + factor * f_top

        if scalar_info.values:
            r = r[0]

        res[target_idx] = r

    return res


# values and p have a different shape, p is 1D and target is 1D
def _to_level_1(data, src_coord, nlev, coord_target, interpolation, scalar_info, xp, res):
    # The coordinate levels must be ordered in descending order with respect to the
    # first dimension. So index 0 has the highest coordinate values, index -1 the lowest,
    # as if it were pressure levels in the atmosphere. The algorithm below agnostic to the
    # actual meaning of the coordinate in the real atmosphere. The terms "top" and "bottom"
    # are used with respect to this coordinate ordering in mind and not related to actual
    # vertical position in the atmosphere. Of course, if the  coordinate is pressure these
    # two definitions coincide.

    # initialize the output array
    res = xp.empty((len(coord_target),) + data.shape[1:], dtype=data.dtype)

    # print("src_coord", src_coord, compare)

    # p on a level is a number
    for target_idx, tc in enumerate(coord_target):
        # initialise the output array
        r = xp.empty(data.shape[1:])
        r = xp.atleast_1d(r)

        # find the level below the target
        idx_bottom = (src_coord > tc).sum(0)

        # print(f"tc: {tc} i_top: {i_top}", src_coord[0])
        if idx_bottom == 0:
            if xp.isclose(src_coord[0], tc):
                r = data[0]
                # print(f"tc: {tc} r: {r}")
            else:
                r.fill(xp.nan)
        elif idx_bottom == nlev:
            if xp.isclose(src_coord[-1], tc):
                r = data[-1]
            else:
                r.fill(xp.nan)
        else:
            top = idx_bottom
            bottom = idx_bottom - 1

            c_top = src_coord[top]
            c_bottom = src_coord[bottom]

            d_top = data[top]
            d_bottom = data[bottom]
            # calculate the interpolation factor
            if interpolation == "linear":
                factor = (tc - c_bottom) / (c_top - c_bottom)
            elif interpolation == "log":
                factor = (xp.log(tc) - xp.log(c_bottom)) / (xp.log(c_top) - xp.log(c_bottom))

            r = (1.0 - factor) * d_bottom + factor * d_top

        res[target_idx] = r

    return res
