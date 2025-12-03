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

import numpy as np
from earthkit.utils.array import array_namespace
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from earthkit.meteo import constants

ScalarInfo = namedtuple("ScalarInfo", ["values", "source", "target"])


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


def relative_geopotential_thickness(
    alpha: NDArray[Any], delta: NDArray[Any], t: NDArray[Any], q: NDArray[Any]
) -> NDArray[Any]:
    """Calculate the geopotential thickness with respect to the surface on model full-levels.

    Parameters
    ----------
    alpha : array-like
        alpha term of pressure calculations
    delta : array-like
        delta term of pressure calculations
    t : array-like
        specific humidity on model full-levels (kg/kg).  First dimension must
        correspond to the model full-levels.
    q : array-like
        temperature on model full-levels (K).  First dimension must
        correspond to the model full-levels.

    Returns
    -------
    array-like
        geopotential thickness of model full-levels with respect to the surface

    Notes
    -----
    ``t`` and ``q`` must contain the same model levels in ascending order with respect to
    the model level number. The model level range must be contiguous and must include the
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


def interpolate_to_pressure_levels(
    data: ArrayLike,
    p_data: Union[ArrayLike, list, tuple, float, int],
    p_target: Union[ArrayLike, list, tuple, float, int],
    interpolation: str = "linear",
) -> ArrayLike:
    """Interpolate data from source to target pressure levels.

    Parameters
    ----------
    data : array-like
        Data to be interpolated. First dimension must correspond to the pressure levels. Must have at
        least two levels. Levels must be ordered in ascending or descending order.
    p_data : array-like
        Pressure levels corresponding to the first dimension of ``data``. Either must have
        the same shape as ``data`` or be a 1D array with length equal to the size of the first
        dimension of ``data``. The units are in Pa.
    p_target : array-like
        Target pressure levels to which ``data`` will be interpolated (Pa). It can be either a scalar
        or a 1D array of pressure levels. Alternatively, it can be an array of arrays where each sub-array
        contains the target pressure levels for the corresponding horizontal location in ``data``. The
        units are in Pa.
    interpolation  : str, optional
        Interpolation mode. Default is "linear". Possible values are:
        - "linear": linear interpolation in pressure
        - "log": linear interpolation in logarithm of pressure
        - "nearest": nearest neighbour interpolation

    Returns
    -------
    array-like
        Data interpolated to the target pressure levels. The shape depends on the shape of ``target_pressure``:
        - If ``target_pressure`` is a scalar, the output shape is equal to ``data.shape[1:]``.
        - If ``target_pressure`` is a 1D array of length N, the output shape is (N, ) + ``data.shape[1:]``.
        - If ``p_target`` is an array of arrays, the output shape is (M, ) + ``data.shape[1:]``,
          where M is the number of horizontal locations in ``data``.
        When interpolation is not possible for a given target pressure level (e.g., when the target pressure
        is outside the range of ``p_data``), the corresponding output values are set to NaN.

    Raises
    ------
    ValueError
        If ``data`` has less than two levels.
    ValueError
        If the first dimension of ``data`` and that of ``p_data`` do not match.


    Notes
    -----

    - The ordering of the input pressure levels is not checked.
    - The units of ``p_data`` and ``p_target`` are assumed to be the Pa; no checks
      or conversions are performed.

    """

    return interpolate_to_coord_levels(data, p_data, p_target, positive="down", interpolation=interpolation)


def interpolate_to_height_levels(
    data: ArrayLike,
    h_data: Union[ArrayLike, list, tuple, float, int],
    h_target: Union[ArrayLike, list, tuple, float, int],
    interpolation: str = "linear",
) -> ArrayLike:
    """Interpolate data from source to target height levels.

    Parameters
    ----------
    data : array-like
        Data to be interpolated. First dimension must correspond to the pressure levels. Must have at
        least two levels. Levels must be ordered in ascending or descending order.
    h_data : array-like
        Height levels corresponding to the first dimension of ``data``. Either must have
        the same shape as ``data`` or be a 1D array with length equal to the size of the first
        dimension of ``data``. The units are in meters.
    h_target : array-like
        Target height levels to which ``data`` will be interpolated (m). It can be either a scalar
        or a 1D array of height levels. Alternatively, it can be an array of arrays where each sub-array
        contains the target height levels for the corresponding horizontal location in ``data``. The
        units are in meters.
    interpolation : str, optional
        Interpolation mode. Default is "linear". Possible values are:
        - "linear": linear interpolation in height
        - "log": linear interpolation in logarithm of height
        - "nearest": nearest neighbour interpolation

    Returns
    -------
    array-like
        Data interpolated to the target height levels. The shape depends on the shape of ``h_target``:
        - If ``h_target`` is a scalar, the output shape is equal to ``data.shape[1:]``.
        - If ``h_target`` is a 1D array of length N, the output shape is (N, ) + ``data.shape[1:]``.
        - If ``h_target`` is an array of arrays, the output shape is (M, ) + ``data.shape[1:]``,
          where M is the number of horizontal locations in ``data``.
        When interpolation is not possible for a given target height level (e.g., when the target height
        is outside the range of ``h_data``), the corresponding output values are set to NaN.

    Raises
    ------
    ValueError
        If ``data`` has less than two levels.
    ValueError
        If the first dimension of ``data`` and that of ``h_data`` do not match.


    Notes
    -----

    - The ordering of the input height levels is not checked.
    - The units of ``h_data`` and ``h_target`` are assumed to be meters; no checks
      or conversions are performed.

    """

    return interpolate_to_coord_levels(data, h_data, h_target, "up", interpolation=interpolation)


def interpolate_to_coord_levels(
    data: ArrayLike,
    coord_data: Union[ArrayLike, list, tuple, float, int],
    coord_target: Union[ArrayLike, list, tuple, float, int],
    positive: str = "down",
    interpolation: str = "linear",
) -> ArrayLike:
    xp = array_namespace(data, coord_data)
    coord_target = xp.atleast_1d(coord_target)
    coord_data = xp.atleast_1d(coord_data)

    if positive not in ["down", "up"]:
        raise ValueError(f"Unknown value for 'positive': {positive}. Allowed values are 'down' and 'up'.")

    if positive == "up":
        compare = xp.less
    else:
        compare = xp.greater

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
        if scalar_info.target:
            return _to_level_1(
                data, coord_data, nlev, coord_target, interpolation, scalar_info, xp, res, compare
            )
        else:
            coord_data = xp.broadcast_to(coord_data, (nlev,) + data.shape[1:]).T

    return _to_level(data, coord_data, nlev, coord_target, interpolation, scalar_info, xp, res, compare)


# values and p have the same shape
def _to_level(data, src_coord, nlev, coord_target, interpolation, scalar_info, xp, res, compare):
    for target_idx, tc in enumerate(coord_target):
        # find the level above the target pressure
        # i_top = (src_coord > tc).sum(0)
        i_top = (compare(src_coord, tc)).sum(0)
        i_top = xp.atleast_1d(i_top)

        # initialise the output array
        r = xp.empty(i_top.shape)

        # mask when the target pressure is below the lowest level
        mask_bottom = i_top == 0
        if xp.any(mask_bottom):
            if interpolation != "nearest":
                r[mask_bottom] = np.nan
                m = mask_bottom & (src_coord[0] == tc)
                r[m] = data[0][m]

        # mask when the target pressure is above the highest level
        mask_top = i_top == nlev
        if xp.any(mask_top):
            if interpolation != "nearest":
                r[mask_top] = np.nan
                m = mask_top & (src_coord[-1] == tc)
                r[m] = data[-1][m]

        # mask when the target pressure is between the lowest and highest levels
        mask_mid = ~(mask_bottom | mask_top)

        if any(mask_mid):
            i_lev = i_top
            indices = np.indices(i_lev.shape)
            masked_indices = tuple(dim[mask_mid] for dim in indices)
            top = (i_top[mask_mid],) + masked_indices
            bottom = (i_top[mask_mid] - 1,) + masked_indices

            c_top = src_coord[top]
            c_bottom = src_coord[bottom]

            f_top = data[top]
            f_bottom = data[bottom]

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
def _to_level_1(data, src_coord, nlev, coord_target, interpolation, scalar_info, xp, res, compare):
    # initialize the output array
    res = xp.empty((len(coord_target),) + data.shape[1:], dtype=data.dtype)

    # p on a level is a number
    for target_idx, tc in enumerate(coord_target):
        # initialise the output array
        r = xp.empty(data.shape[1:])
        r = xp.atleast_1d(r)

        # find the level above the target pressure
        i_top = (compare(src_coord, tc)).sum(0)
        if i_top == 0:
            if xp.isclose(src_coord[0], tc):
                r = data[0]
            else:
                r.fill(xp.nan)
        elif i_top == nlev:
            if xp.isclose(src_coord[-1], tc):
                r = data[-1]
            else:
                r.fill(xp.nan)
        else:
            top = i_top
            bottom = i_top - 1

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
