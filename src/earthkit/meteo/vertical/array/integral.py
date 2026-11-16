# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import Any

from earthkit.utils.array import array_namespace
from numpy.typing import ArrayLike

from earthkit.meteo import constants


def _vertical_array(xp: Any, arr: ArrayLike, data_shape: tuple, vertical_dim: int, name: str) -> ArrayLike:
    """Return ``arr`` with the vertical axis first and the dimensionality of the data.

    ``data_shape`` is the shape of the data with its vertical axis moved to the first
    position. ``arr`` is either 1D, in which case trailing singleton dimensions are
    added to get the same number of dimensions as the data, or it has the same number of
    dimensions as the data with the vertical axis at ``vertical_dim`` and all the other
    dimensions singleton or matching those of the data.
    """
    ndim = len(data_shape)
    arr = xp.asarray(arr)
    shape = arr.shape
    if arr.ndim == 1:
        return xp.reshape(arr, (shape[0],) + (1,) * (ndim - 1))

    if arr.ndim != ndim:
        raise ValueError(
            f'"{name}" must be 1D or have the same number of dimensions as "data" ({ndim}). Found {name}.shape={shape}.'
        )

    arr = xp.moveaxis(arr, vertical_dim, 0)
    if any(s != 1 and s != d for s, d in zip(arr.shape[1:], data_shape[1:])):
        raise ValueError(
            f'"{name}" with shape {shape} is not compatible with "data" with shape {tuple(data_shape)}. '
            f'The vertical axis must be at "vertical_dim"={vertical_dim} and all the other dimensions '
            "must be singleton or match those of the data."
        )
    return arr


def _vertical_weights(xp: Any, x: ArrayLike, data_shape: tuple, vertical_dim: int, name: str) -> ArrayLike:
    """Compute the vertical integration weights of the coordinate ``x`` along the first axis.

    ``data_shape`` is the shape of the data with the vertical axis first and ``N =
    data_shape[0]`` levels. ``x`` holds the coordinate values either on the data levels
    (``N`` values) or on the layer interfaces (``N + 1`` values), as a 1D array or in the
    shape of the data (see :func:`_vertical_array`).

    With level coordinates the trapezoidal rule is used: the weight of level ``i`` is
    ``abs(x[i + 1] - x[i - 1]) / 2`` with half weights at the ends. With interface
    coordinates the weight of level ``k`` is the exact thickness of the layer between
    interfaces ``k`` and ``k + 1``: ``abs(x[k + 1] - x[k])``. The absolute differences
    make the direction of the coordinate irrelevant, but ``x`` must be monotonic along
    the vertical. The weights are returned with the vertical axis first and the same
    number of dimensions as the data.
    """
    nlev = data_shape[0]
    x = _vertical_array(xp, x, data_shape, vertical_dim, name)

    if x.shape[0] == nlev + 1:  # interface coordinates
        return xp.abs(x[1:, ...] - x[:-1, ...])

    if x.shape[0] != nlev:
        raise ValueError(
            f'Inconsistent number of levels: "{name}" has {x.shape[0]} value(s) along the vertical axis, '
            f"but data has {nlev} level(s). It must contain {nlev} values (on the data levels) or "
            f"{nlev + 1} values (on the layer interfaces)."
        )

    if nlev < 2:
        raise ValueError(
            f'At least two levels are needed to integrate over "{name}" given on the data levels. Found '
            f"{nlev} level(s). Pass the {nlev + 1} layer interface values instead."
        )

    mid = 0.5 * (x[:-1, ...] + x[1:, ...])
    left = xp.concatenate((x[:1, ...], mid), axis=0)
    right = xp.concatenate((mid, x[-1:, ...]), axis=0)
    return xp.abs(right - left)


def vertical_integral_pressure(data: ArrayLike, p: ArrayLike, vertical_dim: int = 0) -> ArrayLike:
    r"""Integrate data over pressure.

    *New in version 1.4.0*

    Computes the integral of ``data`` along the axis defined by ``vertical_dim``:

    .. math::

        I = \frac{1}{g} \int_{p_{top}}^{p_{bot}} f \, dp

    where :math:`f` are the data values, :math:`p` is the pressure (Pa) and :math:`g` is
    the gravitational acceleration on the surface of the Earth (see
    :py:attr:`earthkit.meteo.constants.g`). All quantities are in SI units.

    Parameters
    ----------
    data : ArrayLike
        Data to be integrated. The axis corresponding to the vertical coordinate is
        defined by ``vertical_dim``.
    p : ArrayLike
        Pressure (Pa), either on the data levels (:math:`N` values along the vertical)
        or on the layer interfaces (:math:`N + 1` values), where :math:`N` is the number
        of vertical levels of ``data``. Either a 1D array or an array with the same
        number of dimensions as ``data`` with the vertical axis at ``vertical_dim`` and
        all the other dimensions singleton or matching those of ``data``. When interfaces
        are given, the data level :math:`k` lies between interface :math:`k` and
        interface :math:`k + 1`, in the same order as ``data``. The values must be
        monotonic along the vertical axis, in either direction; they are neither checked
        nor reordered, because the absolute differences make the direction irrelevant.
        See Notes for the integration schemes.
    vertical_dim : int
        Axis corresponding to the vertical coordinate in ``data`` and in ``p``. Default
        is 0 (first axis).

    Returns
    -------
    ArrayLike
        The vertically integrated data. The shape is the shape of ``data`` with the
        vertical axis removed. NaN values in the input propagate to the output.

    Raises
    ------
    ValueError
        If the number of vertical values of ``p`` is neither :math:`N` nor :math:`N + 1`,
        if the shape of ``p`` is incompatible with the shape of ``data`` or if the level
        coordinates contain fewer than two levels.

    See Also
    --------
    vertical_integral_height
    pressure_on_hybrid_levels

    Notes
    -----
    With the hydrostatic relation, :math:`dp = -\rho g \, dz` (:math:`\rho` is the air
    density), the integral above is the mass-weighted integral :math:`\int \rho f \, dz`.
    The :math:`1/g` form is therefore valid only for a pressure coordinate under
    hydrostatic balance. Otherwise use :func:`vertical_integral_height` with the density
    given in ``rho``.

    When ``p`` contains the :math:`N` level values the trapezoidal rule is used between
    the first and the last level:

    .. math::

        I = \frac{1}{g} \sum_{i=1}^{N} w_i f_i \ , \qquad w_i = \frac{| p_{i+1} - p_{i-1} |}{2}

    with the half weights :math:`w_1 = | p_2 - p_1 | / 2` and
    :math:`w_N = | p_N - p_{N-1} | / 2` at the ends. It requires :math:`N \geq 2`. The
    column is not extended to the surface or to the top of the atmosphere: only the layer
    between the outermost levels is integrated.

    When ``p`` contains the :math:`N + 1` interface values the exact layer thicknesses
    are used:

    .. math::

        I = \frac{1}{g} \sum_{k=1}^{N} f_k \, | p_{k+1/2} - p_{k-1/2} |

    This form is exact for layer-constant values (or when ``data`` holds layer means),
    not for arbitrary fields. It is the natural form for the IFS hybrid (model) levels.
    The interface pressures of all the levels are computed with::

        p_half = pressure_on_hybrid_levels(sp, A, B, output="half")

    and the ones enveloping a subset of the full levels with::

        _, p_half = pressure_on_hybrid_levels(sp, A, B, levels=levels, output=("full", "half"))

    which returns the :math:`N + 1` interfaces ordered like ``levels`` (e.g. full levels
    ``[2, 3, 4]`` give interfaces ``[1, 2, 3, 4]`` and full levels ``[4, 3, 2]`` give
    ``[4, 3, 2, 1]``). Note that ``levels=..., output="half"`` interprets ``levels`` as
    half-level numbers and returns only :math:`N` values. Both outputs have the vertical
    axis first, as the default ``vertical_dim=0`` expects.
    """
    xp = array_namespace(data, p)
    data = xp.moveaxis(xp.asarray(data), vertical_dim, 0)
    w = _vertical_weights(xp, p, data.shape, vertical_dim, "p")
    return xp.sum(data * w, axis=0) / constants.g


def vertical_integral_height(
    data: ArrayLike, h: ArrayLike, rho: ArrayLike | None = None, vertical_dim: int = 0
) -> ArrayLike:
    r"""Integrate data over geometric height.

    *New in version 1.4.0*

    Computes the integral of ``data`` along the axis defined by ``vertical_dim``.
    Without ``rho``:

    .. math::

        I = \int_{z_{bot}}^{z_{top}} f \, dz

    and with ``rho`` the mass-weighted integral:

    .. math::

        I = \int_{z_{bot}}^{z_{top}} \rho f \, dz

    where :math:`f` are the data values, :math:`z` is the geometric height (m) and
    :math:`\rho` is the air density (kg m-3). All quantities are in SI units.

    Parameters
    ----------
    data : ArrayLike
        Data to be integrated. The axis corresponding to the vertical coordinate is
        defined by ``vertical_dim``.
    h : ArrayLike
        Geometric height (m), either on the data levels (:math:`N` values along the
        vertical) or on the layer interfaces (:math:`N + 1` values), where :math:`N` is
        the number of vertical levels of ``data``. Either a 1D array or an array with the
        same number of dimensions as ``data`` with the vertical axis at ``vertical_dim``
        and all the other dimensions singleton or matching those of ``data``. When
        interfaces are given, the data level :math:`k` lies between interface :math:`k`
        and interface :math:`k + 1`, in the same order as ``data``. The values must be
        monotonic along the vertical axis, in either direction; they are neither checked
        nor reordered, because the absolute differences make the direction irrelevant.
        See Notes for the integration schemes.
    rho : ArrayLike | None
        Air density (kg m-3) on the data levels: a 1D array with :math:`N` values or an
        array in the shape of ``data``, defined the same way as ``h`` but only on the
        :math:`N` data levels. When None (default) the plain integral :math:`\int f dz`
        is computed.
    vertical_dim : int
        Axis corresponding to the vertical coordinate in ``data``, ``h`` and ``rho``.
        Default is 0 (first axis).

    Returns
    -------
    ArrayLike
        The vertically integrated data. The shape is the shape of ``data`` with the
        vertical axis removed. NaN values in the input propagate to the output.

    Raises
    ------
    ValueError
        If the number of vertical values of ``h`` is neither :math:`N` nor :math:`N + 1`,
        if ``rho`` is not defined on the :math:`N` data levels, if the shape of ``h`` or
        ``rho`` is incompatible with the shape of ``data`` or if the level coordinates
        contain fewer than two levels.

    See Also
    --------
    vertical_integral_pressure

    Notes
    -----
    No hydrostatic assumption is made and the gravitational acceleration is not used:
    the mass-weighted integral is computed directly from the density given in ``rho`` by
    applying the weights described below to the product :math:`\rho f`. With interface
    coordinates the result is exact only when :math:`\rho` and ``data`` are constant
    within each layer, or when their product :math:`\rho_k f_k` equals the layer mean of
    :math:`\rho f`. Separate layer means of :math:`\rho` and :math:`f` are not enough,
    because the mean of a product generally differs from the product of the means.

    When ``h`` contains the :math:`N` level values the trapezoidal rule is used between
    the first and the last level:

    .. math::

        I = \sum_{i=1}^{N} w_i f_i \ , \qquad w_i = \frac{| h_{i+1} - h_{i-1} |}{2}

    with the half weights :math:`w_1 = | h_2 - h_1 | / 2` and
    :math:`w_N = | h_N - h_{N-1} | / 2` at the ends. It requires :math:`N \geq 2`. The
    column is not extended to the surface or to the top of the atmosphere: only the layer
    between the outermost levels is integrated.

    When ``h`` contains the :math:`N + 1` interface values the exact layer thicknesses
    are used:

    .. math::

        I = \sum_{k=1}^{N} f_k \, | h_{k+1/2} - h_{k-1/2} |

    This form is exact for layer-constant values (or, without ``rho``, when ``data`` holds
    layer means), not for arbitrary fields.
    """
    xp = array_namespace(data, h, rho)
    data = xp.moveaxis(xp.asarray(data), vertical_dim, 0)
    w = _vertical_weights(xp, h, data.shape, vertical_dim, "h")

    if rho is not None:
        rho = _vertical_array(xp, rho, data.shape, vertical_dim, "rho")
        if rho.shape[0] != data.shape[0]:
            raise ValueError(
                f'Inconsistent number of levels: "rho" has {rho.shape[0]} value(s) along the vertical axis, '
                f"but data has {data.shape[0]} level(s). It must be defined on the data levels."
            )
        data = data * rho

    return xp.sum(data * w, axis=0)
