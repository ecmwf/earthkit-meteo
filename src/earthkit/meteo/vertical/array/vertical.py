# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from typing import Tuple
from typing import Union

import deprecation
import numpy as np
from earthkit.utils.array import array_namespace
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from earthkit.meteo import constants


@deprecation.deprecated(deprecated_in="0.7", details="Use pressure_on_hybrid_levels instead.")
def pressure_at_model_levels(
    A: NDArray[Any], B: NDArray[Any], sp: Union[float, NDArray[Any]], alpha_top: str = "ifs"
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    r"""Compute pressure at model full- and half-levels.

    *Deprecated in version 0.7.0*
    See :ref:`deprecated-hybrid-pressure-at-model-levels` for details.

    Parameters
    ----------
    A : ndarray
        A-coefficients defining the model levels. See [IFS-CY47R3-Dynamics]_
        Chapter 2, Section 2.2.1. for details.
    B : ndarray
        B-coefficients defining the model levels. See [IFS-CY47R3-Dynamics]_
        Chapter 2, Section 2.2.1. for details.
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
        Alpha at full-levels


    Notes
    -----
    ``A`` and ``B`` must contain the same model half-levels in ascending order with
    respect to the model level number. The model level range must be contiguous and
    must include the bottom-most model half-level (surface), but not all the levels
    must be present. E.g. if the vertical coordinate system has 137 model levels using
    only a subset of levels between e.g. 137-96 is allowed.

    For details on the returned parameters see [IFS-CY47R3-Dynamics]_ Chapter 2, Section 2.2.1.

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

    A = np.asarray(A)
    B = np.asarray(B)

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

    # pressure at highest half-level <= 0.1
    if np.any(p_half_level[0, ...] <= PRESSURE_TOA):
        delta[0, ...] = np.log(p_half_level[1, ...] / PRESSURE_TOA)
    # pressure at highest half-level > 0.1
    else:
        delta[0, ...] = np.log(p_half_level[1, ...] / p_half_level[0, ...])

    # calculate alpha
    alpha = np.zeros(new_shape_full)

    alpha[1:, ...] = (
        1.0 - p_half_level[1:-1, ...] / (p_half_level[2:, ...] - p_half_level[1:-1, ...]) * delta[1:, ...]
    )

    # pressure at highest half-level <= 0.1
    if np.any(p_half_level[0, ...] <= PRESSURE_TOA):
        alpha[0, ...] = alpha_top
    # pressure at highest half-level > 0.1
    else:
        alpha[0, ...] = (
            1.0 - p_half_level[0, ...] / (p_half_level[1, ...] - p_half_level[0, ...]) * delta[0, ...]
        )

    # calculate pressure on model full-levels
    # TODO: is there a faster way to calculate the averages?
    # TODO: introduce option to calculate full-levels in more complicated way
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

    *Deprecated in version 0.7.0*
    See :ref:`deprecated-hybrid-relative-geopotential-thickness` for details.

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

    The computations are described in [IFS-CY47R3-Dynamics]_ Chapter 2, Section 2.2.1.

    See also
    --------
    pressure_at_model_levels

    """
    from earthkit.meteo.thermo import specific_gas_constant

    xp = array_namespace(alpha, delta, q, t)

    R = specific_gas_constant(q)
    d = R * t

    # compute geopotential thickness on half-levels from 1 to NLEV-1
    dphi_half = xp.cumulative_sum(xp.flip(d[1:, ...] * delta[1:, ...], axis=0), axis=0)
    dphi_half = xp.flip(dphi_half, axis=0)

    # compute geopotential thickness on full-levels
    dphi = xp.zeros_like(d)
    dphi[:-1, ...] = dphi_half + d[:-1, ...] * alpha[:-1, ...]
    dphi[-1, ...] = d[-1, ...] * alpha[-1, ...]

    return dphi


@deprecation.deprecated(deprecated_in="0.7", details="Use interpolate_hybrid_to_height_levels instead.")
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

    *Deprecated in version 0.7.0*
    See :ref:`deprecated-hybrid-pressure-at-height-levels` for details.

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
    A = np.asarray(A)
    B = np.asarray(B)

    # geopotential thickness of the height level
    tdphi = height * constants.g

    nlev = A.shape[0] - 1  # number of model full-levels

    # pressure(-related) variables
    p_full, p_half, delta, alpha = pressure_at_model_levels(A, B, sp, alpha_top=alpha_top)

    # relative geopotential thickness of full-levels
    dphi = relative_geopotential_thickness(alpha, delta, t, q)

    # find the model full-level right above the height level
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
    A: ArrayLike,
    B: ArrayLike,
    sp: ArrayLike,
    levels=None,
    alpha_top="ifs",
    output="full",
    vertical_axis=0,
) -> ArrayLike:
    r"""Compute pressure and related parameters on hybrid (IFS model) levels.

    *New in version 0.7.0*: This function replaces the deprecated :func:`pressure_at_model_levels`.

    Parameters
    ----------
    A : array-like
        A-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number (from the top of the
        atmosphere toward the surface). If the total number of (full) model levels
        is :math:`NLEV`, ``A`` must contain :math:`NLEV+1` values, one for each
        half-level. See [IFS-CY47R3-Dynamics]_ Chapter 2, Section 2.2.1. for
        details.
    B : array-like
        B-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number. Must have the same
        size and ordering as ``A``.
        See [IFS-CY47R3-Dynamics]_ Chapter 2, Section 2.2.1. for details.
    sp : array-like
        Surface pressure (Pa)
    levels : None, array-like, list, tuple, optional
        Specify the hybrid full-levels to return in the given order. Following the
        IFS convention model level numbering starts at 1 at the top of the atmosphere
        and increasing toward the surface.  If None (default), all the levels are
        returned in the order defined by the A and B coefficients (i.e. ascending order
        with respect to the model level number).
    alpha_top : str, optional
        Option to initialise the alpha parameters (for details see below) on the top of the
        model atmosphere (first half-level in the vertical coordinate system). The possible
        values are:

        - "ifs": alpha is set to log(2). See [IFS-CY47R3-Dynamics]_ Chapter 2, Section 2.2.1. for details.
        - "arpege": alpha is set to 1.0

    output : str or list/tuple of str, optional
        Specify which outputs to return. Possible values are "full", "half", "delta" and "alpha".
        Can be a single string or a list/tuple of strings. Default is "full". The outputs are:

        - "full": pressure (Pa) on full-levels
        - "half": pressure (Pa) on half-levels. When ``levels`` is None, returns all the
          half-levels. When ``levels`` is not None, only returns the half-levels below
          the requested full-levels.
        - "delta": logarithm of pressure difference between two adjacent half-levels. Uses
          the same indexing as the full-levels.
        - "alpha": alpha parameter defined for layers (i.e. for full-levels). Uses the same
          indexing as the full-levels. Used for the calculation of the relative geopotential
          thickness on full-levels. See
          :func:`relative_geopotential_thickness_on_hybrid_levels` for details.

    vertical_axis : int, optional
        Axis corresponding to the vertical coordinate (hybrid levels) in the output arrays.
        Default is 0 (first axis).

    Returns
    -------
    array-like or tuple of array-like
        See the ``output`` parameter for details. The axis corresponding to the vertical
        coordinate (hybrid levels) in the output arrays is defined by the ``vertical_axis``
        parameter.

    Notes
    -----
    The hybrid model levels divide the atmosphere into :math:`NLEV` layers. These layers are defined
    by the pressures at the interfaces between them for :math:`0 \leq k \leq NLEV`, which are
    the half-levels :math:`p_{k+1/2}` (indices increase from the top of the atmosphere towards
    the surface). The half-levels are defined by the ``A`` and ``B`` coefficients in such a way
    that at the top of the atmosphere the first half-level pressure :math:`p_{+1/2}` is a constant,
    while at the surface :math:`p_{NLEV+1/2}` is the surface pressure.

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

    For more details see [IFS-CY47R3-Dynamics]_ Chapter 2, Section 2.2.1.

    Examples
    --------
    - :ref:`/examples/hybrid_levels.ipynb`


    See also
    --------
    relative_geopotential_thickness_on_hybrid_levels

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

    xp = array_namespace(sp, A, B)
    A = xp.asarray(A)
    B = xp.asarray(B)
    device = xp.device(sp)

    if levels is not None:
        # select a contiguous subset of levels
        nlev = A.shape[0] - 1  # number of model full-levels
        levels = xp.asarray(levels)
        levels_max = int(levels.max())
        levels_min = int(levels.min())
        if levels_max > nlev:
            raise ValueError(f"Requested level {levels_max} exceeds the maximum number of levels {nlev}.")
        if levels_min < 1:
            raise ValueError(f"Level numbering starts at 1. Found level={levels_min} < 1.")

        half_idx = xp.asarray(list(range(levels_min - 1, levels_max + 1)))
        A = A[half_idx]
        B = B[half_idx]

        # compute indices to select the requested full-levels later
        # out_half_idx = xp.where(levels[:, None] == half_idx[None, :])[1]

        out_half_idx = xp.nonzero(xp.asarray(levels[:, None] == half_idx[None, :]))[1]

        out_full_idx = out_half_idx - 1

    # make the calculation agnostic to the number of dimensions
    ndim = sp.ndim
    new_shape_half = (A.shape[0],) + (1,) * ndim
    A_reshaped = xp.reshape(A, new_shape_half)
    B_reshaped = xp.reshape(B, new_shape_half)

    # calculate pressure on model half-levels
    p_half_level = A_reshaped + B_reshaped * sp[xp.newaxis, ...]

    if "delta" in output or "alpha" in output:
        # constants
        PRESSURE_TOA = 0.1  # safety when highest pressure level = 0.0

        alpha_top = np.log(2) if alpha_top == "ifs" else 1.0

        new_shape_full = (A.shape[0] - 1,) + sp.shape

        # calculate delta
        delta = xp.zeros(new_shape_full, device=device)
        delta[1:, ...] = xp.log(p_half_level[2:, ...] / p_half_level[1:-1, ...])

        # pressure at highest half-level<= 0.1
        if xp.any(p_half_level[0, ...] <= PRESSURE_TOA):
            delta[0, ...] = xp.log(p_half_level[1, ...] / PRESSURE_TOA)
        # pressure at highest half-level > 0.1
        else:
            delta[0, ...] = xp.log(p_half_level[1, ...] / p_half_level[0, ...])

        # calculate alpha
        alpha = xp.zeros(new_shape_full, device=device)

        alpha[1:, ...] = (
            1.0 - p_half_level[1:-1, ...] / (p_half_level[2:, ...] - p_half_level[1:-1, ...]) * delta[1:, ...]
        )

        # pressure at highest half-level <= 0.1
        if xp.any(p_half_level[0, ...] <= PRESSURE_TOA):
            alpha[0, ...] = alpha_top
        # pressure at highest half-level > 0.1
        else:
            alpha[0, ...] = (
                1.0 - p_half_level[0, ...] / (p_half_level[1, ...] - p_half_level[0, ...]) * delta[0, ...]
            )

    if "full" in output:
        # calculate pressure on model full-levels
        # TODO: is there a faster way to calculate the averages?
        # TODO: introduce option to calculate full-levels in more complicated way
        # p_full_level = xp.apply_along_axis(
        #     lambda m: xp.convolve(m, xp.ones(2) / 2, mode="valid"), axis=0, arr=p_half_level
        # )

        p_full_level = p_half_level[:-1, ...] + 0.5 * xp.diff(p_half_level, axis=0)

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

    if vertical_axis != 0 and res[0].ndim > 1:
        # move the vertical axis to the required position
        res = [xp.moveaxis(r, 0, vertical_axis) for r in res]

    if len(res) == 1:
        return res[0]

    return tuple(res)


def _compute_relative_geopotential_thickness_on_hybrid_levels(
    t: ArrayLike,
    q: ArrayLike,
    alpha: ArrayLike,
    delta: ArrayLike,
    xp: Any,
) -> ArrayLike:
    """Compute the geopotential thickness between the surface and hybrid (IFS model) full-levels.

    *New in version 0.7.0*: This function replaces the deprecated :func:`relative_geopotential_thickness`.

    Parameters
    ----------
    t : array-like
        Temperature on hybrid full-levels (K). The axis corresponding to the vertical
        coordinate (hybrid levels) is defined by the ``vertical_axis`` parameter.
        The levels must be in ascending order with respect the model level number. Not
        all the levels must be present, but a contiguous level range including the bottom-most
        level must be used. E.g. if the vertical coordinate system has 137 model levels using
        only a subset of levels between e.g. 137-96 is allowed.
    q : array-like
        Specific humidity on hybrid full-levels (kg/kg). Must have the
        same shape, level range and order as ``t``.
    alpha : array-like
        Alpha term of pressure calculations computed using
        :func:`pressure_on_hybrid_levels`. Must have the same shape, level range
        and order as ``t``.
    delta : array-like
        Delta term of pressure calculations computed using :func:`pressure_on_hybrid_levels`.
        Must have the same shape, level range and order as ``t``.
    vertical_axis : int, optional
        Axis corresponding to the vertical coordinate (hybrid levels) in the input arrays
        and also in the output array. Default is 0 (first axis).

    Returns
    -------
    array-like
        Geopotential thickness (m2/s2) between the surface and hybrid full-levels.
        The axis corresponding to the vertical coordinate (hybrid levels) is defined
        by the ``vertical_axis`` parameter.

    Notes
    -----
    ``alpha`` and ``delta``can be calculated using :func:`pressure_on_hybrid_levels`.

    The computations are described in [IFS-CY47R3-Dynamics]_ Chapter 2, Section 2.2.1.

    Examples
    --------
    - :ref:`/examples/hybrid_levels.ipynb`


    See also
    --------
    pressure_on_hybrid_levels

    """
    from earthkit.meteo.thermo import specific_gas_constant

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


def relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta(
    t: ArrayLike,
    q: ArrayLike,
    alpha: ArrayLike,
    delta: ArrayLike,
    vertical_axis=0,
) -> ArrayLike:
    """Compute the geopotential thickness between the surface and hybrid full-levels (IFS model levels).

    *New in version 0.7.0*: This function replaces the deprecated :func:`relative_geopotential_thickness`.

    Parameters
    ----------
    t : array-like
        Temperature on hybrid full-levels (K). The axis corresponding to the vertical
        coordinate (hybrid levels) is defined by the ``vertical_axis`` parameter.
        The levels must be in ascending order with respect the model level number. Not
        all the levels must be present, but a contiguous level range including the bottom-most
        level must be used. E.g. if the vertical coordinate system has 137 model levels using
        only a subset of levels between e.g. 137-96 is allowed.
    q : array-like
        Specific humidity on hybrid full-levels (kg/kg). Must have the
        same shape, level range and order as ``t``.
    alpha : array-like
        Alpha term of pressure calculations computed using
        :func:`pressure_on_hybrid_levels`. Must have the same shape, level range
        and order as ``t``.
    delta : array-like
        Delta term of pressure calculations computed using :func:`pressure_on_hybrid_levels`.
        Must have the same shape, level range and order as ``t``.
    vertical_axis : int, optional
        Axis corresponding to the vertical coordinate (hybrid levels) in the input arrays
        and also in the output array. Default is 0 (first axis).

    Returns
    -------
    array-like
        Geopotential thickness (m2/s2) between the surface and hybrid full-levels.
        The axis corresponding to the vertical coordinate (hybrid levels) is defined
        by the ``vertical_axis`` parameter.

    Notes
    -----
    ``alpha`` and ``delta`` can be calculated using :func:`pressure_on_hybrid_levels`.
    The computations are described in [IFS-CY47R3-Dynamics]_ Chapter 2, Section 2.2.1.

    Examples
    --------
    - :ref:`/examples/hybrid_levels.ipynb`


    See also
    --------
    pressure_on_hybrid_levels

    """

    xp = array_namespace(alpha, delta, q, t)
    alpha = xp.asarray(alpha)
    delta = xp.asarray(delta)
    t = xp.asarray(t)
    q = xp.asarray(q)

    if vertical_axis != 0:
        # move the vertical axis to the first position
        alpha = xp.moveaxis(alpha, vertical_axis, 0)
        delta = xp.moveaxis(delta, vertical_axis, 0)
        t = xp.moveaxis(t, vertical_axis, 0)
        q = xp.moveaxis(q, vertical_axis, 0)

    dphi = _compute_relative_geopotential_thickness_on_hybrid_levels(t, q, alpha, delta, xp)

    if vertical_axis != 0:
        # move the vertical axis back to its original position
        dphi = xp.moveaxis(dphi, 0, vertical_axis)

    return dphi


def relative_geopotential_thickness_on_hybrid_levels(
    t: ArrayLike,
    q: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    sp: ArrayLike,
    alpha_top="ifs",
    vertical_axis=0,
) -> ArrayLike:
    """Compute the geopotential thickness between the surface and hybrid full-levels (IFS model levels).

    *New in version 0.7.0*

    Parameters
    ----------
    t : array-like
        Temperature on hybrid full-levels (K). The axis corresponding to the vertical
        coordinate (hybrid levels) is defined by the ``vertical_axis`` parameter.
        The levels must be in ascending order with respect the model level number. Not
        all the levels must be present, but a contiguous level range including the bottom-most
        level must be used. E.g. if the vertical coordinate system has 137 model levels using
        only a subset of levels between e.g. 137-96 is allowed.
    q : array-like
        Specific humidity on hybrid full-levels (kg/kg). Must have the
        same shape, level range and order as ``t``.
    A : array-like
        A-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number.
    B : array-like
        B-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number.  Must have the same
        size as ``A``.
    sp : array-like
        Surface pressure (Pa)
    alpha_top : str, optional
        Option to initialise the alpha parameters (for details see below) on the top of the
        model atmosphere (first half-level in the vertical coordinate system). See
        :func:`pressure_on_hybrid_levels` for details.
    vertical_axis : int, optional
        Axis corresponding to the vertical coordinate (model levels) in the input ``t``
        and ``q`` arrays and also in the output array. Default is 0 (first axis).

    Returns
    -------
    array-like
        Geopotential thickness (m2/s2) between the surface and hybrid full-levels. The
        axis corresponding to the vertical coordinate (hybrid levels) is defined by the
        ``vertical_axis`` parameter.

    Notes
    -----
    The computations are done in two steps:

    - first the ``alpha`` and ``delta`` parameters are calculated using
      :func:`pressure_on_hybrid_levels`
    - then the geopotential thickness is calculated with hydrostatic integration using
      :func:`relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta` See
      [IFS-CY47R3-Dynamics]_ Chapter 2, Section 2.2.1. for details.

    Examples
    --------
    - :ref:`/examples/hybrid_levels.ipynb`

    See also
    --------
    pressure_on_hybrid_levels
    relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta

    """
    xp = array_namespace(t, q, A, B, sp)
    A = xp.asarray(A)
    B = xp.asarray(B)
    sp = xp.asarray(sp)
    t = xp.asarray(t)
    q = xp.asarray(q)

    levels = _hybrid_subset(t, A, B, vertical_axis)

    alpha, delta = pressure_on_hybrid_levels(
        A, B, sp, alpha_top=alpha_top, levels=levels, output=("alpha", "delta")
    )

    # return relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta(
    #     t, q, alpha, delta, vertical_axis=vertical_axis
    # )

    # move the vertical axis to the first position
    if vertical_axis != 0:
        # move the vertical axis to the first position
        alpha = xp.moveaxis(alpha, vertical_axis, 0)
        delta = xp.moveaxis(delta, vertical_axis, 0)
        t = xp.moveaxis(t, vertical_axis, 0)
        q = xp.moveaxis(q, vertical_axis, 0)

    dphi = _compute_relative_geopotential_thickness_on_hybrid_levels(t, q, alpha, delta, xp)

    # move the vertical axis back to its original position
    if vertical_axis != 0:
        dphi = xp.moveaxis(dphi, 0, vertical_axis)

    return dphi


def geopotential_on_hybrid_levels(
    t: ArrayLike,
    q: ArrayLike,
    zs: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    sp: ArrayLike,
    alpha_top="ifs",
    vertical_axis=0,
):
    """Compute the geopotential on hybrid (IFS model) full-levels.

    *New in version 0.7.0*

    Parameters
    ----------
    t : array-like
        Temperature on hybrid full-levels (K). The axis corresponding to the vertical
        coordinate (hybrid levels) is defined by the ``vertical_axis`` parameter.
        The levels must be in ascending order with respect the model level number. Not
        all the levels must be present, but a contiguous level range including the bottom-most
        level must be used. E.g. if the vertical coordinate system has 137 model levels using
        only a subset of levels between e.g. 137-96 is allowed.
    q : Specific humidity on hybrid full-levels (kg/kg). Must have the
        same shape, level range and order as ``t``.
    zs : array-like
        Surface geopotential (m2/s2). Only used when ``reference_level`` is "sea".
    A : array-like
        A-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number.
    B : array-like
        B-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number. Must have the same
        size as ``A``.
    sp : array-like
        Surface pressure (Pa)
    alpha_top : str, optional
        Option to initialise the alpha parameters (for details see below) on the top of the
        model atmosphere (first half-level in the vertical coordinate system). See
        :func:`pressure_on_hybrid_levels` for details.
    vertical_axis : int, optional
        Axis corresponding to the vertical coordinate (model levels) in the input ``t``
        and ``q`` arrays and also in the output array. Default is 0 (first axis).


    Returns
    -------
    array-like
        Geopotential (m2/s2) on hybrid full-levels. The axis corresponding to the vertical
        coordinate (hybrid levels) is defined by the ``vertical_axis`` parameter.

    Notes
    -----
    The computations are described in [IFS-CY47R3-Dynamics]_ Chapter 2, Section 2.2.1.


    Examples
    --------
    - :ref:`/examples/hybrid_levels.ipynb`


    See also
    --------
    pressure_on_hybrid_levels
    relative_geopotential_thickness_on_hybrid_levels

    """
    z = relative_geopotential_thickness_on_hybrid_levels(
        t, q, A, B, sp, vertical_axis=vertical_axis, alpha_top=alpha_top
    )
    xp = array_namespace(z, zs)
    zs = xp.asarray(zs)
    return z + zs


def height_on_hybrid_levels(
    t: ArrayLike,
    q: ArrayLike,
    zs: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    sp: ArrayLike,
    alpha_top="ifs",
    h_type: str = "geometric",
    h_reference: str = "ground",
    vertical_axis=0,
):
    """Compute the height on hybrid (IFS model) full-levels.

    *New in version 0.7.0*

    Parameters
    ----------
    t : array-like
        Temperature on hybrid full-levels (K). The axis corresponding to the vertical
        coordinate (hybrid levels) is defined by the ``vertical_axis`` parameter.
        The levels must be in ascending order with respect the model level number. Not
        all the levels must be present, but a contiguous level range including the bottom-most
        level must be used. E.g. if the vertical coordinate system has 137 model levels using
        only a subset of levels between e.g. 137-96 is allowed.
    q : array-like
        Specific humidity on hybrid full-levels (kg/kg). Must have the
        same shape, level range and order as ``t``.
    zs : array-like
        Surface geopotential (m2/s2). Not used  when ``h_type`` is "geopotential" and
        ``h_reference`` is "ground".
    A : array-like
        A-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number.
    B : array-like
        B-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number. Must have the same
        size as ``A``.
    sp : array-like
        Surface pressure (Pa)
    alpha_top : str, optional
        Option to initialise the alpha parameters (for details see below) on the top of the
        model atmosphere (first half-level in the vertical coordinate system). See
        :func:`pressure_on_hybrid_levels` for details.
    h_type : str, optional
        Type of height to compute. Default is "geometric". Possible values are:

        - "geometric": geometric height (m) with respect to ``h_reference``
        - "geopotential": geopotential height (m) with respect to ``h_reference``

        See :func:`geometric_height_from_geopotential` and
        :func:`geopotential_height_from_geopotential` for details.

    h_reference : str, optional
        Reference level for the height calculation. Default is "ground". Possible values are:

        - "ground": height with respect to the ground/surface level
        - "sea": height with respect to the sea level

    vertical_axis : int, optional
        Axis corresponding to the vertical coordinate (hybrid full-levels) in the input
        arrays and also in the output array. Default is 0 (first axis).

    Returns
    -------
    array-like
        Height (m) of hybrid full-levels with
        respect to ``h_reference``. The type of height is defined by ``h_type``
        ("geometric" or "geopotential"). The axis corresponding to the vertical
        coordinate (hybrid levels) is defined by the ``vertical_axis`` parameter.

    Notes
    -----
    The height is calculated from the geopotential on hybrid levels, which is computed
    from the ``t``, ``q``, ``zs`` and the hybrid
    level definition (``A``, ``B``  and ``sp``). The
    computations are described in [IFS-CY47R3-Dynamics]_ Chapter 2, Section 2.2.1.


    Examples
    --------
    - :ref:`/examples/hybrid_levels.ipynb`

    See also
    --------
    hybrid_level_parameters
    pressure_on_hybrid_levels
    geopotential_on_hybrid_levels
    relative_geopotential_thickness_on_hybrid_levels
    """

    if h_reference not in ["sea", "ground"]:
        raise ValueError(f"Unknown '{h_reference=}'. Use 'sea' or 'ground'.")

    z_thickness = relative_geopotential_thickness_on_hybrid_levels(
        t, q, A, B, sp, alpha_top=alpha_top, vertical_axis=vertical_axis
    )

    xp = array_namespace(z_thickness)

    if h_reference == "sea":
        zs = xp.asarray(zs)
        z = z_thickness + zs
        if h_type == "geometric":
            h = geometric_height_from_geopotential(z)
        else:
            h = geopotential_height_from_geopotential(z)
    else:
        if h_type == "geometric":
            zs = xp.asarray(zs)
            h_surf = geometric_height_from_geopotential(zs)
            z = z_thickness + zs
            h = geometric_height_from_geopotential(z) - h_surf
        else:
            h = geopotential_height_from_geopotential(z_thickness)

    return h


def _hybrid_subset(data, A, B, vertical_axis=0):
    """Helper function to determine the subset of hybrid levels corresponding to the data levels."""
    nlev_t = data.shape[vertical_axis]
    nlev = A.shape[0] - 1  # number of model full-levels
    levels = None
    if nlev_t != nlev:
        # select a contiguous subset of levels
        levels = list(range(nlev - nlev_t + 1, nlev + 1))
        assert nlev_t == len(levels), (
            "Inconsistent number of levels between data and A/B coefficients."
            f" data have {nlev_t} levels, A/B have {nlev} levels."
        )
    return levels


def interpolate_hybrid_to_pressure_levels(
    data: ArrayLike,
    target_p: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    sp: ArrayLike,
    alpha_top="ifs",
    interpolation: str = "linear",
    aux_bottom_data=None,
    aux_bottom_p=None,
    aux_top_data=None,
    aux_top_p=None,
    vertical_axis=0,
):
    """Interpolate data from hybrid full-levels (IFS model levels) to pressure levels.

    *New in version 0.7.0*

    Parameters
    ----------
    data : array-like
        Data to be interpolated. The axis corresponding to the vertical
        coordinate (hybrid levels) is defined by the ``vertical_axis`` parameter.
        Must have at least two levels. Levels must be ordered in ascending order
        with respect to the model level number. By convention, model level numbering
        starts at 1 at the top of the atmosphere and increases towards the surface.
        Not all the levels must be present, but a contiguous level range including the
        bottom-most level must be used. E.g. if the vertical coordinate system has 137 model
        levels using only a subset of levels between e.g. 137-96 is allowed.
    target_p : array-like
        Target pressure levels (Pa) to which ``data`` will be interpolated. It can be
        either a scalar or a 1D array of pressure levels. Alternatively, it can be a
        multidimensional array with a vertical axis defined by ``vertical_axis``. In this
        case the other axes/dimensions must match those of ``data``.
    A : array-like
        A-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number.
        See :func:`hybrid_level_parameters` for details.
    B : array-like
        B-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number. Must have the same
        size as ``A``. See :func:`hybrid_level_parameters` for details.
    sp : array-like
        Surface pressure (Pa). The shape must be compatible with the non-vertical
        dimensions of ``data``.
    alpha_top : str, optional
        Option to initialise the alpha parameters on the top of the
        model atmosphere (first half-level in the vertical coordinate system). See
        :func:`pressure_on_hybrid_levels` for details.
    interpolation  : str, optional
        Interpolation mode. Default is "linear". Possible values are:

        - "linear": linear interpolation in pressure between the two nearest levels
        - "log": linear interpolation in logarithm of pressure between the two nearest levels
        - "nearest": nearest level interpolation

    aux_bottom_data : array-like, optional
        Auxiliary data for interpolation to targets below the bottom hybrid full-level
        and above the level specified by ``aux_bottom_p``. Can be a scalar or must have the
        same shape as a single level of ``data``.
    aux_bottom_p : array-like, optional
        Pressures (Pa) of ``aux_bottom_data``. Can be a scalar or must have the same
        shape as a single level of ``data``.
    aux_top_data : array-like, optional
        Auxiliary data for interpolation to targets above the top hybrid full-level
        and below the level specified by ``aux_top_p``. Can be a scalar or must have
        the same shape as a single level of ``data``.
    aux_top_p : array-like, optional
        Pressures (Pa) of ``aux_top_data``. Can be a scalar or must have the same
        shape as a single level of ``data``.
    vertical_axis : int, optional
        Axis corresponding to the vertical coordinate (hybrid full-levels) in the input
        arrays and also in the output array. Default is 0 (first axis).


    Returns
    -------
    array-like
        Data interpolated to the target levels. The shape depends on the shape of ``target_p``.
        The axis corresponding to the vertical coordinate (hybrid levels) is defined by
        the ``vertical_axis`` parameter. When interpolation is not possible for a given target
        pressure level (e.g., when the target pressure is outside the available pressure range),
        the corresponding output values are set to nan.

    Raises
    ------
    ValueError
        If ``data`` has less than two levels.
    ValueError
        If the first dimension of ``data`` and that of ``target_p`` do not match.


    Examples
    --------
    - :ref:`/examples/interpolate_hybrid_to_pl.ipynb`

    See also
    --------
    interpolate_monotonic

    """
    xp = array_namespace(data, A, B, sp)
    data = xp.asarray(data)
    A = xp.asarray(A)
    B = xp.asarray(B)
    sp = xp.asarray(sp)

    levels = _hybrid_subset(data, A, B, vertical_axis)

    p = pressure_on_hybrid_levels(A, B, sp, alpha_top=alpha_top, levels=levels, output="full")
    return interpolate_monotonic(
        data=data,
        coord=p,
        target_coord=target_p,
        interpolation=interpolation,
        aux_min_level_coord=aux_top_p,
        aux_min_level_data=aux_top_data,
        aux_max_level_coord=aux_bottom_p,
        aux_max_level_data=aux_bottom_data,
        vertical_axis=vertical_axis,
    )


def interpolate_hybrid_to_height_levels(
    data: ArrayLike,
    target_h: ArrayLike,
    t: ArrayLike,
    q: ArrayLike,
    zs: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    sp: ArrayLike,
    alpha_top="ifs",
    h_type: str = "geometric",
    h_reference: str = "ground",
    interpolation: str = "linear",
    aux_bottom_data=None,
    aux_bottom_h=None,
    aux_top_data=None,
    aux_top_h=None,
    vertical_axis=0,
):
    """Interpolate data from hybrid full-levels (IFS model levels) to height levels.

    *New in version 0.7.0*

    Parameters
    ----------
    data : array-like
        Data to be interpolated. The axis corresponding to the vertical
        coordinate (hybrid levels) is defined by the ``vertical_axis`` parameter.
        Must have at least two levels. Levels must be ordered in ascending order
        with respect to the model level number.  By convention, model level numbering
        starts at 1 at the top of the atmosphere and increases towards the surface.  Not
        all the levels must be present, but a contiguous level range
        including the bottom-most level must be used. E.g. if the vertical coordinate
        system has 137 model levels using only a subset of levels between
        e.g. 137-96 is allowed.
    target_h : array-like
        Target height levels (m) to which ``data`` will be interpolated. It can be
        either a scalar or a 1D array of height levels. Alternatively, it can be a
        multidimensional array with a vertical axis defined by `vertical_axis`. In this case
        the other axes/dimensions must match those of ``data``. The type of the height and
        the reference level are defined by ``h_type`` and ``h_reference``.
    t : array-like
        Temperature on hybrid full-levels (K). Must have the
        same shape, level range and order as ``data``.
    q : array-like
        Specific humidity on hybrid full-levels (kg/kg). Must have the
        same shape, level range and order as ``t``.
    zs : array-like
        Surface geopotential (m2/s2). The shape
        must be compatible with the non-vertical dimensions of ``t`` and ``q``.
        Not used  when ``h_type`` is "geopotential" and ``h_reference`` is "ground".
    A : array-like
        A-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number.
        See :func:`hybrid_level_parameters` for details.
    B : array-like
        B-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number. Must have the same
        size as ``A``. See :func:`hybrid_level_parameters` for details.
    sp : array-like
        Surface pressure (Pa). The shape must be compatible with the non-vertical
        dimensions of ``data``.
    alpha_top : str, optional
        Option to initialise the alpha parameters (for details see below) on the top of the
        model atmosphere (first half-level in the vertical coordinate system). See
        :func:`pressure_on_hybrid_levels` for details.
    h_type : str, optional
        Type of height to compute. Default is "geometric".  Possible values are:

        - "geometric": geometric height (m) with respect to ``h_reference``
        - "geopotential": geopotential height (m) with respect to ``h_reference``

        See :func:`geometric_height_from_geopotential` and
        :func:`geopotential_height_from_geopotential` for details.
    h_reference : str, optional
        Reference level for the height calculation. Default is "ground". Possible values are:

        - "ground": height with respect to the ground/surface level
        - "sea": height with respect to the sea level

    interpolation  : str, optional
        Interpolation mode. Default is "linear". Possible values are:

        - "linear": linear interpolation in height between the two nearest levels
        - "log": linear interpolation in logarithm of height between the two nearest levels
        - "nearest": nearest level interpolation

    aux_bottom_data : array-like, optional
        Auxiliary data for interpolation to heights between the bottom hybrid full-level
        and ``aux_bottom_h``. Can be a scalar or must have
        the same shape as a single level of ``data``.
    aux_bottom_h : array-like, optional
        Heights (m) of ``aux_bottom_data``. Can be a scalar or must have the same
        shape as a single level of ``data``.  The type of the height and
        the reference level are defined by ``h_type`` and ``h_reference``.
    aux_top_data : array-like, optional
        Auxiliary data for interpolation to heights above the top hybrid full-level
        and below ``aux_top_h``. Can be a scalar or must have
        the same shape as a single level of ``data``.
    aux_top_h : array-like, optional
        Heights (m) of ``aux_top_data``. Can be a scalar or must have the same
        shape as a single level of ``data``.  The type of the height and
        the reference level are defined by ``h_type`` and ``h_reference``.
    vertical_axis : int, optional
        Axis corresponding to the vertical coordinate (hybrid full-levels) in the input
        arrays and also in the output array. Default is 0 (first axis).


    Returns
    -------
    array-like
        Data interpolated to the target height levels. The shape depends on the shape
        of ``target_h``. The axis corresponding to the vertical coordinate (hybrid levels)
        is defined by the ``vertical_axis`` parameter. When interpolation is not possible
        for a given target height level (e.g., when the target height is outside the
        available height range), the corresponding output values are set to nan.

    Raises
    ------
    ValueError
        If ``data`` has less than two levels.
    ValueError
        If the first dimension of ``data`` and that of ``target_h`` do not match.

    Examples
    --------
    - :ref:`/examples/interpolate_hybrid_to_hl.ipynb`


    See also
    --------
    interpolate_monotonic

    """
    h = height_on_hybrid_levels(
        t,
        q,
        zs,
        A,
        B,
        sp,
        alpha_top=alpha_top,
        h_type=h_type,
        h_reference=h_reference,
        vertical_axis=vertical_axis,
    )

    return interpolate_monotonic(
        data=data,
        coord=h,
        target_coord=target_h,
        interpolation=interpolation,
        aux_min_level_data=aux_bottom_data,
        aux_min_level_coord=aux_bottom_h,
        aux_max_level_coord=aux_top_h,
        aux_max_level_data=aux_top_data,
        vertical_axis=vertical_axis,
    )


def interpolate_pressure_to_height_levels(
    data: ArrayLike,
    target_h: ArrayLike,
    z: ArrayLike,
    zs: ArrayLike,
    h_type: str = "geometric",
    h_reference: str = "ground",
    interpolation: str = "linear",
    aux_bottom_data=None,
    aux_bottom_h=None,
    aux_top_data=None,
    aux_top_h=None,
    vertical_axis: int = 0,
):
    """Interpolate data from pressure levels to height levels.

    *New in version 0.7.0*

    Parameters
    ----------
    data : array-like
        Data to be interpolated. The axis corresponding to the vertical
        coordinate (pressure levels) is defined by the ``vertical_axis`` parameter.
        Must have at least two levels. Levels must be ordered in ascending or
        descending order with respect to pressure (i.e. monotonic).
    target_h : array-like
        Target height levels (m) to which ``data`` will be interpolated. It can be
        either a scalar or a 1D array of height levels. Alternatively, it can be a
        multidimensional array with a vertical axis defined by `vertical_axis`. In this case
        the other axes/dimensions must match those of ``data``. The type of the height and
        the reference level are defined by ``h_type`` and ``h_reference``.
    z : array-like
        Geopotential (m2/s2) on the same pressure levels as ``data``.
    zs : array-like
        Surface geopotential (m2/s2). The shape must be compatible with the non-vertical
        dimensions of ``data`` and ``z``. Only used when and ``h_reference`` is "ground".
    h_type : str, optional
        Type of height to compute. Possible values are:

        - "geometric": geometric height (m) with respect to ``h_reference``
        - "geopotential": geopotential height (m) with respect to ``h_reference``
          Default is "geometric". See :func:`geometric_height_from_geopotential` and
          :func:`geopotential_height_from_geopotential` for details.

    h_reference : str, optional
        Reference level for the height calculation. Default is "ground". Possible values are:

        - "ground": height with respect to the ground/surface level
        - "sea": height with respect to the sea level

    interpolation  : str, optional
        Interpolation mode. Default is "linear". Possible values are:

        - "linear": linear interpolation in height between the two nearest levels
        - "log": linear interpolation in logarithm of height between the two nearest levels
        - "nearest": nearest level interpolation

    aux_bottom_data : array-like, optional
        Auxiliary data for interpolation to heights between the bottom pressure
        level and ``aux_bottom_h``. Can be a scalar or must have
        the same shape as a single level of ``data``.
    aux_bottom_h : array-like, optional
        Heights (m) of ``aux_bottom_data``. Can be a scalar or must have the same
        shape as a single level of ``data``. The type of the height and
        the reference level are defined by ``h_type`` and ``h_reference``.
    aux_top_data : array-like, optional
        Auxiliary data for interpolation to heights between the top pressure
        level and ``aux_top_h``. Can be a scalar or must have
        the same shape as a single level of ``data``.
    aux_top_h : array-like, optional
        Heights (m) of ``aux_top_data``. Can be a scalar or must have the same
        shape as a single level of ``data``. The type of the height and
        the reference level are defined by ``h_type`` and ``h_reference``.
    vertical_axis : int, optional
        Axis corresponding to the vertical coordinate (hybrid full-levels) in the input
        arrays and also in the output array. Default is 0 (first axis).


    Returns
    -------
    array-like
        Data interpolated to the target height levels. The shape depends on the shape
        of ``target_h``. The axis corresponding to the vertical coordinate (height levels)
        is defined by the ``vertical_axis`` parameter. When interpolation is not possible
        for a given target height level (e.g., when the target height is outside the
        available height range), the corresponding output values are set to nan.

    Raises
    ------
    ValueError
        If ``data`` has less than two levels.
    ValueError
        If the first dimension of ``data`` and that of ``target_h`` do not match.

    Examples
    --------
    - :ref:`/examples/interpolate_pl_to_hl.ipynb`


    See also
    --------
    interpolate_monotonic

    """
    if h_type == "geometric":
        h = geometric_height_from_geopotential(z)
        if h_reference == "ground":
            zs_h = geometric_height_from_geopotential(zs)
            h = h - zs_h
    else:
        if h_reference == "ground":
            z = z - zs
        h = geopotential_height_from_geopotential(z)

    return interpolate_monotonic(
        data=data,
        coord=h,
        target_coord=target_h,
        interpolation=interpolation,
        aux_min_level_data=aux_bottom_data,
        aux_min_level_coord=aux_bottom_h,
        aux_max_level_coord=aux_top_h,
        aux_max_level_data=aux_top_data,
        vertical_axis=vertical_axis,
    )


def interpolate_monotonic(
    data: ArrayLike,
    coord: Union[ArrayLike, list, tuple, float, int],
    target_coord: Union[ArrayLike, list, tuple, float, int],
    interpolation: str = "linear",
    aux_min_level_data=None,
    aux_min_level_coord=None,
    aux_max_level_data=None,
    aux_max_level_coord=None,
    vertical_axis: int = 0,
) -> ArrayLike:
    """Interpolate data between the same type of monotonic coordinate levels.

    *New in version 0.7.0*

    Parameters
    ----------
    data : array-like
        Data to be interpolated. The axis corresponding to the vertical
        coordinate is defined by the ``vertical_axis`` parameter.
        Must have at least two levels.
    coord : array-like
        Vertical coordinates related to ``data``. Either must have the same
        shape as ``data`` or be a 1D array with length equal to the size of
        the number of levels in ``data``. Must be monotonic (either sorted
        ascending or descending) along the vertical axis.
    target_coord : array-like
        Target coordinate levels to which ``data`` will be interpolated. It can be
        either a scalar or a 1D array of coordinate levels. Alternatively, it can be a
        multidimensional array with a vertical axis defined by `vertical_axis`. In this case
        the other axes/dimensions must match those of ``data``. Must be the same type
        of coordinate as ``coord``.
    interpolation  : str, optional
        Interpolation mode. Default is "linear". Possible values are:

        - "linear": linear interpolation in coordinate between the two nearest levels
        - "log": linear interpolation in logarithm of coordinate between the two nearest levels
        - "nearest": nearest level interpolation

    aux_min_level_data : array-like, optional
        Auxiliary data for interpolation to target levels below the minimum level
        of ``coord`` and above `aux_min_level_coord`. Can be a scalar or must have
        the same shape as a single level of ``data``.
    aux_min_level_coord : array-like, optional
        Coordinates of ``aux_min_level_data``. Can be a scalar or must have the same
        shape as a single level of ``data`` or ``coord``. Must be the same type
        of coordinate as ``coord``.
    aux_max_level_data : array-like, optional
        Auxiliary data for interpolation to target levels above the maximum level
        of ``coord`` and below `aux_max_level_coord`. Can be a scalar or must have
        the same shape as a single level of ``data``.
    aux_max_level_coord : array-like, optional
        Coordinates of ``aux_max_level_data``. Can be a scalar or must have the
        same shape as a single level of ``data`` or ``coord``. Must be the same type
        of coordinate as ``coord``.
    vertical_axis : int, optional
        Axis corresponding to the vertical coordinate in the input arrays and also in the
        output array. Default is 0 (first axis).


    Returns
    -------
    array-like
        Data interpolated to the target levels. The shape depends on the shape of ``target_coord``.
        The axis corresponding to the vertical coordinate is defined by
        the ``vertical_axis`` parameter. When interpolation is not possible for a given target
        level (e.g., when the target level is outside the available level range),
        the corresponding output values are set to nan.

    Raises
    ------
    ValueError
        If ``data`` has less than two levels.
    ValueError
        If the shape of ``data`` and that of ``coord`` are not compatible.

    Notes
    -----
    - The ordering of the input coordinate levels is not checked.
    - The units of ``coord`` and ``target_coord`` are assumed to be the same; no checks
      or conversions are performed.

    Examples
    --------
    - :ref:`/examples/interpolate_hybrid_to_pl.ipynb`
    - :ref:`/examples/interpolate_hybrid_to_hl.ipynb`
    - :ref:`/examples/interpolate_pl_to_hl.ipynb`
    - :ref:`/examples/interpolate_pl_to_pl.ipynb`

    """
    from .monotonic import MonotonicInterpolator

    comp = MonotonicInterpolator()
    return comp(
        data,
        coord,
        target_coord,
        interpolation,
        aux_min_level_data,
        aux_min_level_coord,
        aux_max_level_data,
        aux_max_level_coord,
        vertical_axis=vertical_axis,
    )
