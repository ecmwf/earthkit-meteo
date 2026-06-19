# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,  # noqa: F401
    Literal,
    TypeAlias,
    overload,
)

from earthkit.meteo import constants
from earthkit.meteo.utils.decorators import dispatch

if TYPE_CHECKING:
    import xarray  # type: ignore[import]
    from earthkit.data import Field, FieldList  # type: ignore[import]

ArrayLike: TypeAlias = Any


@overload
def geopotential_height_from_geopotential(z: "ArrayLike") -> "ArrayLike": ...


@overload
def geopotential_height_from_geopotential(z: "xarray.DataArray") -> "xarray.DataArray": ...


@overload
def geopotential_height_from_geopotential(z: "FieldList") -> "FieldList": ...


def geopotential_height_from_geopotential(
    z: "ArrayLike | xarray.DataArray | FieldList",
) -> "ArrayLike | xarray.DataArray | FieldList":
    r"""Compute geopotential height from geopotential.

    Parameters
    ----------
    z : array-like | xarray.DataArray | FieldList
        Geopotential (m2/s2).

    Returns
    -------
    array-like | xarray.DataArray | FieldList
        Geopotential height (m).


    The computation is based on the following definition:

    .. math::

        gh = \frac{z}{g}

    where :math:`g` is the gravitational acceleration on the surface of the Earth
    (see :data:`earthkit.meteo.constants.g`).


    Implementations
    ---------------
    :func:`geopotential_height_from_geopotential` calls one of the following implementations
    depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.geopotential_height_from_geopotential` for array-like
    - :py:meth:`earthkit.meteo.vertical.xarray.geopotential_height_from_geopotential` for xarray.DataArray
    - :py:meth:`earthkit.meteo.vertical.fieldlist.geopotential_height_from_geopotential` for FieldList

    The function returns an object of the same type as the input arguments.
    """
    return dispatch(geopotential_height_from_geopotential, array=True)(z)


@overload
def geopotential_from_geopotential_height(gh: "ArrayLike") -> "ArrayLike": ...


@overload
def geopotential_from_geopotential_height(gh: "xarray.DataArray") -> "xarray.DataArray": ...


@overload
def geopotential_from_geopotential_height(gh: "FieldList") -> "FieldList": ...


def geopotential_from_geopotential_height(
    gh: "ArrayLike | xarray.DataArray | FieldList",
) -> "ArrayLike | xarray.DataArray | FieldList":
    r"""Compute geopotential from geopotential height.

    Parameters
    ----------
    gh : array-like | xarray.DataArray | FieldList
        Geopotential height (m).

    Returns
    -------
    array-like | xarray.DataArray | FieldList
        Geopotential (m2/s2).


    The computation is based on the following definition:

    .. math::

        z = gh \cdot g

    where :math:`g` is the gravitational acceleration on the surface of the Earth
    (see :data:`earthkit.meteo.constants.g`).


    Implementations
    ---------------
    :func:`geopotential_from_geopotential_height` calls one of the following implementations
    depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.geopotential_from_geopotential_height` for array-like
    - :py:meth:`earthkit.meteo.vertical.xarray.geopotential_from_geopotential_height` for xarray.DataArray
    - :py:meth:`earthkit.meteo.vertical.fieldlist.geopotential_from_geopotential_height` for FieldList

    The function returns an object of the same type as the input arguments.
    """
    return dispatch(geopotential_from_geopotential_height, array=True)(gh)


@overload
def geopotential_height_from_geometric_height(
    h: "ArrayLike",
    R_earth: float = constants.R_earth,
) -> "ArrayLike": ...


@overload
def geopotential_height_from_geometric_height(
    h: "xarray.DataArray",
    R_earth: float = constants.R_earth,
) -> "xarray.DataArray": ...


@overload
def geopotential_height_from_geometric_height(
    h: "FieldList",
    R_earth: float = constants.R_earth,
) -> "FieldList": ...


def geopotential_height_from_geometric_height(
    h: "ArrayLike | xarray.DataArray | FieldList",
    R_earth: float = constants.R_earth,
) -> "ArrayLike | xarray.DataArray | FieldList":
    r"""Compute the geopotential height from geometric height.

    Parameters
    ----------
    h : array-like | xarray.DataArray | FieldList
        Geometric height with respect to sea level (m).
    R_earth : float, optional
        Average radius of the Earth (m). Default is :data:`earthkit.meteo.constants.R_earth`.

    Returns
    -------
    array-like | xarray.DataArray | FieldList
        Geopotential height (m).


    The computation is based on the following formula:

    .. math::

        gh = \frac{h \cdot R_{earth}}{R_{earth} + h}

    where :math:`R_{earth}` is the average radius of the Earth
    (see :data:`earthkit.meteo.constants.R_earth`).


    Implementations
    ---------------
    :func:`geopotential_height_from_geometric_height` calls one of the following implementations
    depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.geopotential_height_from_geometric_height` for array-like
    - :py:meth:`earthkit.meteo.vertical.xarray.geopotential_height_from_geometric_height` for xarray.DataArray
    - :py:meth:`earthkit.meteo.vertical.fieldlist.geopotential_height_from_geometric_height` for FieldList

    The function returns an object of the same type as the input arguments.
    """
    return dispatch(geopotential_height_from_geometric_height, array=True)(h, R_earth=R_earth)


@overload
def geopotential_from_geometric_height(
    h: "ArrayLike",
    R_earth: float = constants.R_earth,
) -> "ArrayLike": ...


@overload
def geopotential_from_geometric_height(
    h: "xarray.DataArray",
    R_earth: float = constants.R_earth,
) -> "xarray.DataArray": ...


@overload
def geopotential_from_geometric_height(
    h: "FieldList",
    R_earth: float = constants.R_earth,
) -> "FieldList": ...


def geopotential_from_geometric_height(
    h: "ArrayLike | xarray.DataArray | FieldList",
    R_earth: float = constants.R_earth,
) -> "ArrayLike | xarray.DataArray | FieldList":
    r"""Compute the geopotential from geometric height.

    Parameters
    ----------
    h : array-like | xarray.DataArray | FieldList
        Geometric height with respect to sea level (m).
    R_earth : float, optional
        Average radius of the Earth (m). Default is :data:`earthkit.meteo.constants.R_earth`.

    Returns
    -------
    array-like | xarray.DataArray | FieldList
        Geopotential (m2/s2).


    The computation is based on the following formula:

    .. math::

        z = \frac{h \cdot g \cdot R_{earth}}{R_{earth} + h}

    where

    - :math:`R_{earth}` is the average radius of the Earth
      (see :data:`earthkit.meteo.constants.R_earth`)
    - :math:`g` is the gravitational acceleration on the surface of the Earth
      (see :data:`earthkit.meteo.constants.g`)


    Implementations
    ---------------
    :func:`geopotential_from_geometric_height` calls one of the following implementations
    depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.geopotential_from_geometric_height` for array-like
    - :py:meth:`earthkit.meteo.vertical.xarray.geopotential_from_geometric_height` for xarray.DataArray
    - :py:meth:`earthkit.meteo.vertical.fieldlist.geopotential_from_geometric_height` for FieldList

    The function returns an object of the same type as the input arguments.
    """
    return dispatch(geopotential_from_geometric_height, array=True)(h, R_earth=R_earth)


@overload
def geometric_height_from_geopotential_height(
    gh: "ArrayLike",
    R_earth: float = constants.R_earth,
) -> "ArrayLike": ...


@overload
def geometric_height_from_geopotential_height(
    gh: "xarray.DataArray",
    R_earth: float = constants.R_earth,
) -> "xarray.DataArray": ...


@overload
def geometric_height_from_geopotential_height(
    gh: "FieldList",
    R_earth: float = constants.R_earth,
) -> "FieldList": ...


def geometric_height_from_geopotential_height(
    gh: "ArrayLike | xarray.DataArray | FieldList",
    R_earth: float = constants.R_earth,
) -> "ArrayLike | xarray.DataArray | FieldList":
    r"""Compute the geometric height from geopotential height.

    Parameters
    ----------
    gh : array-like | xarray.DataArray | FieldList
        Geopotential height (m).
    R_earth : float, optional
        Average radius of the Earth (m). Default is :data:`earthkit.meteo.constants.R_earth`.

    Returns
    -------
    array-like | xarray.DataArray | FieldList
        Geometric height (m).


    The computation is based on the following formula:

    .. math::

        h = \frac{R_{earth} \cdot gh}{R_{earth} - gh}

    where :math:`R_{earth}` is the average radius of the Earth
    (see :data:`earthkit.meteo.constants.R_earth`).


    Implementations
    ---------------
    :func:`geometric_height_from_geopotential_height` calls one of the following implementations
    depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.geometric_height_from_geopotential_height` for array-like
    - :py:meth:`earthkit.meteo.vertical.xarray.geometric_height_from_geopotential_height` for xarray.DataArray
    - :py:meth:`earthkit.meteo.vertical.fieldlist.geometric_height_from_geopotential_height` for FieldList

    The function returns an object of the same type as the input arguments.
    """
    return dispatch(geometric_height_from_geopotential_height, array=True)(gh, R_earth=R_earth)


@overload
def geometric_height_from_geopotential(
    z: "ArrayLike",
    R_earth: float = constants.R_earth,
) -> "ArrayLike": ...


@overload
def geometric_height_from_geopotential(
    z: "xarray.DataArray",
    R_earth: float = constants.R_earth,
) -> "xarray.DataArray": ...


@overload
def geometric_height_from_geopotential(
    z: "FieldList",
    R_earth: float = constants.R_earth,
) -> "FieldList": ...


def geometric_height_from_geopotential(
    z: "ArrayLike | xarray.DataArray | FieldList",
    R_earth: float = constants.R_earth,
) -> "ArrayLike | xarray.DataArray | FieldList":
    r"""Compute the geometric height from geopotential.

    Parameters
    ----------
    z : array-like | xarray.DataArray | FieldList
        Geopotential (m2/s2).
    R_earth : float, optional
        Average radius of the Earth (m). Default is :data:`earthkit.meteo.constants.R_earth`.

    Returns
    -------
    array-like | xarray.DataArray | FieldList
        Geometric height (m).


    The computation is based on the following formula:

    .. math::

        h = \frac{R_{earth} \cdot \frac{z}{g}}{R_{earth} - \frac{z}{g}}

    where

    - :math:`R_{earth}` is the average radius of the Earth
      (see :data:`earthkit.meteo.constants.R_earth`)
    - :math:`g` is the gravitational acceleration on the surface of the Earth
      (see :data:`earthkit.meteo.constants.g`)


    Implementations
    ---------------
    :func:`geometric_height_from_geopotential` calls one of the following implementations
    depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.geometric_height_from_geopotential` for array-like
    - :py:meth:`earthkit.meteo.vertical.xarray.geometric_height_from_geopotential` for xarray.DataArray
    - :py:meth:`earthkit.meteo.vertical.fieldlist.geometric_height_from_geopotential` for FieldList

    The function returns an object of the same type as the input arguments.
    """
    return dispatch(geometric_height_from_geopotential, array=True)(z, R_earth=R_earth)


# TODO: figure out to handle this case gracefully
def hybrid_level_parameters(n_levels: int, model: str = "ifs") -> "tuple[ArrayLike, ArrayLike]":
    r"""Get the A and B coefficients of hybrid levels for a given model configuration.

    Parameters
    ----------
    n_levels : int
        Number of (full) hybrid levels. Currently only ``n_levels`` 91 and 137 are supported.
    model : str, optional
        Model name. Default is ``"ifs"``. Currently only ``"ifs"`` is supported.

    Returns
    -------
    tuple[array-like, array-like]
        A tuple ``(A, B)`` of 1D arrays of length ``n_levels + 1`` containing the A and B
        coefficients on the hybrid half-levels.


    Notes
    -----
    The A and B coefficients are not unique; in theory there can be multiple definitions for a
    given number of levels and model. :func:`hybrid_level_parameters` returns the most typical
    set of coefficients used.


    Implementations
    ---------------
    :func:`hybrid_level_parameters` calls the following implementation:

    - :py:meth:`earthkit.meteo.vertical.array.hybrid_level_parameters` for array-like
    """
    from earthkit.meteo.vertical.array import hybrid

    return hybrid.hybrid_level_parameters(n_levels, model=model)


@overload
def pressure_on_hybrid_levels(
    sp: "ArrayLike",
    A: "ArrayLike",
    B: "ArrayLike",
    levels: "ArrayLike | None" = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    output: "Literal['full', 'half', 'delta', 'alpha', 'level'] | list | tuple" = "full",
    vertical_dim: int = 0,
) -> "ArrayLike | tuple[ArrayLike, ...]": ...


@overload
def pressure_on_hybrid_levels(
    sp: "FieldList",
    A: "ArrayLike | None" = None,
    B: "ArrayLike | None" = None,
    levels: "ArrayLike | None" = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    output: "Literal['full', 'half', 'delta', 'alpha'] | list | tuple" = "full",
) -> "FieldList | tuple[FieldList, ...]": ...


def pressure_on_hybrid_levels(
    sp: "ArrayLike | FieldList",
    A: "ArrayLike | None" = None,
    B: "ArrayLike | None" = None,
    levels: "ArrayLike | None" = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    output: "str | list | tuple" = "full",
    vertical_dim: int = 0,
) -> "ArrayLike | tuple[ArrayLike, ...] | FieldList | tuple[FieldList, ...]":
    r"""Compute pressure and related parameters on hybrid (IFS model) levels.

    Parameters
    ----------
    sp : array-like | FieldList
        Surface pressure (Pa). See the concrete implementations for details.
    A : array-like | None
        A-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number (from the top of the
        atmosphere toward the surface). If the total number of (full) model levels
        is :math:`NLEV`, ``A`` must contain :math:`NLEV+1` values, one for each
        half-level. See [IFS-CY49R1-Dynamics]_ Chapter 2, Section 2.2.1. for
        details. Can be None for FieldList/Field input if the coefficients can be
        resolved from the metadata of ``sp``.
    B : array-like | None
        B-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number. Must have the same
        size and ordering as ``A``.
        See [IFS-CY49R1-Dynamics]_ Chapter 2, Section 2.2.1. for details.
        Can be None for FieldList/Field input if the coefficients can be
        resolved from metadata of ``sp``.
    levels : array-like | None, default=None
        Specify the hybrid full-levels to return. Must be contiguous range of levels in
        either ascending or descending order. Following the
        IFS convention model level numbering starts at 1 at the top of the atmosphere
        and increasing toward the surface.  If None (default), all the levels are
        returned in the order defined by the A and B coefficients (i.e. ascending order
        with respect to the model level number). If only half-levels are requested in ``output``
        the ``levels`` are interpreted as half-level numbers (so 0 is a valid half-level
        number corresponding to the top of the atmosphere).
    alpha_top : {"ifs", "arpege"}, default="ifs"
        Option to initialise alpha at the top of the model atmosphere. Possible values are
        ``"ifs"`` (default) and ``"arpege"``.
        See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.
    output : str | list | tuple, default="full"
        Output data to return. See the concrete implementations for details.
    vertical_dim : int, default=0
        Axis corresponding to the vertical coordinate in the output arrays. Default is 0.
        This keyword argument is not supported when the input data is FieldList.

    Returns
    -------
    array-like | tuple[array-like, ...] | FieldList | tuple[FieldList, ...]
        See the concrete implementations for details.


    See Also
    --------
    hybrid_level_parameters
    relative_geopotential_thickness_on_hybrid_levels


    Implementations
    ---------------
    :func:`pressure_on_hybrid_levels` calls one of the following implementations depending on
    the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels` for array-like
    - :py:meth:`earthkit.meteo.vertical.fieldlist.pressure_on_hybrid_levels` for FieldList/Field

    The function returns an object of the same type as the input arguments.
    """
    _kwargs = dict(A=A, B=B, levels=levels, alpha_top=alpha_top, output=output, vertical_dim=vertical_dim)
    if vertical_dim != 0:
        _kwargs["vertical_dim"] = vertical_dim

    return dispatch(pressure_on_hybrid_levels, xarray=False, fieldlist=True, array=True)(sp, **_kwargs)


@overload
def relative_geopotential_thickness_on_hybrid_levels(
    t: "ArrayLike",
    q: "ArrayLike",
    sp: "ArrayLike",
    A: "ArrayLike",
    B: "ArrayLike",
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    vertical_dim: int = 0,
) -> "ArrayLike": ...


@overload
def relative_geopotential_thickness_on_hybrid_levels(
    t: "FieldList",
    q: "FieldList",
    sp: "FieldList",
    A: "ArrayLike | None" = None,
    B: "ArrayLike | None" = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
) -> "FieldList": ...


def relative_geopotential_thickness_on_hybrid_levels(
    t: "ArrayLike | FieldList",
    q: "ArrayLike | FieldList",
    sp: "ArrayLike |  FieldList",
    A: "ArrayLike | None" = None,
    B: "ArrayLike | None" = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    vertical_dim: int = 0,
) -> "ArrayLike | FieldList":
    r"""Compute the geopotential thickness between the surface and hybrid full-levels.

    Parameters
    ----------
    t : array-like | FieldList
        Temperature on hybrid full-levels (K). See the concrete implementations for details.
    q : array-like | FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same shape, level
        range and order as ``t``.
    sp : array-like | FieldList
        Surface pressure (Pa). See the concrete implementations for details.
    A : array-like | None
        A-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number.
        Can be None for FieldList/Field input if the coefficients can be
        resolved from the metadata of ``sp``.
    B : array-like | None, optional
        B-coefficients defining the hybrid levels. Must have the same size as ``A``.
        Can be None for FieldList/Field input if the coefficients can be
        resolved from the metadata of ``sp``.
    alpha_top : {"ifs", "arpege"}, default="ifs"
        Option to initialise alpha at the top of the model atmosphere. Default is ``"ifs"``.
        See :func:`pressure_on_hybrid_levels` for details.
    vertical_dim : int, default=0
        Axis corresponding to the vertical coordinate in the input and output arrays.
        Default is 0. This keyword argument is not supported when the input data is FieldList.

    Returns
    -------
    array-like | FieldList
        Geopotential thickness (m2/s2) between the surface and hybrid full-levels. The axis
        corresponding to the vertical coordinate is defined by ``vertical_dim``.


    See Also
    --------
    pressure_on_hybrid_levels
    relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta


    Implementations
    ---------------
    :func:`relative_geopotential_thickness_on_hybrid_levels` calls one of the following
    implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.relative_geopotential_thickness_on_hybrid_levels`
      for array-like
    - :py:meth:`earthkit.meteo.vertical.fieldlist.relative_geopotential_thickness_on_hybrid_levels`
      for FieldList

    The function returns an object of the same type as the input arguments.
    """
    _kwargs = dict(A=A, B=B, alpha_top=alpha_top)
    if vertical_dim != 0:
        _kwargs["vertical_dim"] = vertical_dim

    return dispatch(relative_geopotential_thickness_on_hybrid_levels, xarray=False, fieldlist=True, array=True)(
        t, q, sp, **_kwargs
    )


@overload
def relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta(
    t: "ArrayLike",
    q: "ArrayLike",
    alpha: "ArrayLike",
    delta: "ArrayLike",
    vertical_dim: int = 0,
) -> "ArrayLike": ...


@overload
def relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta(
    t: "FieldList",
    q: "FieldList",
    alpha: "FieldList",
    delta: "FieldList",
) -> "FieldList": ...


def relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta(
    t: "ArrayLike |  FieldList",
    q: "ArrayLike |  FieldList",
    alpha: "ArrayLike |  FieldList",
    delta: "ArrayLike |  FieldList",
    vertical_dim: int = 0,
) -> "ArrayLike | FieldList":
    r"""Compute the geopotential thickness between the surface and hybrid full-levels from
    pre-computed alpha and delta.

    Parameters
    ----------
    t : ArrayLike |  FieldList
        Temperature on hybrid full-levels (K). See the concrete implementations for details.
    q : ArrayLike |  FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same shape, level
        range and order as ``t``. See the concrete implementations for details.
    alpha : ArrayLike |  FieldList
        Alpha term of pressure calculations computed using :func:`pressure_on_hybrid_levels`.
        Must have the same shape, level range and order as ``t``. See the concrete implementations for details.
    delta : ArrayLike |  FieldList
        Delta term of pressure calculations computed using :func:`pressure_on_hybrid_levels`.
        Must have the same shape, level range and order as ``t``. See the concrete implementations for details.
    vertical_dim : int, optional
        Axis corresponding to the vertical coordinate in the input and output arrays.
        Default is 0. This keyword argument is not supported when the input data is FieldList.

    Returns
    -------
    array-like | FieldList
        Geopotential thickness (m2/s2) between the surface and hybrid full-levels.
        See the concrete implementations for details.


    See Also
    --------
    pressure_on_hybrid_levels
    relative_geopotential_thickness_on_hybrid_levels


    Implementations
    ---------------
    :func:`relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta` calls one of the
    following implementations depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta`
      for array-like
    - :py:meth:`earthkit.meteo.vertical.fieldlist.relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta`
      for FieldList

    The function returns an object of the same type as the input arguments.
    """
    _kwargs = dict()
    if vertical_dim != 0:
        _kwargs["vertical_dim"] = vertical_dim

    return dispatch(
        relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta,
        xarray=False,
        fieldlist=True,
        array=True,
    )(t, q, alpha, delta, **_kwargs)


@overload
def geopotential_on_hybrid_levels(
    t: "ArrayLike",
    q: "ArrayLike",
    zs: "ArrayLike",
    sp: "ArrayLike",
    A: "ArrayLike",
    B: "ArrayLike",
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    vertical_dim: int = 0,
) -> "ArrayLike": ...


@overload
def geopotential_on_hybrid_levels(
    t: "FieldList",
    q: "FieldList",
    zs: "FieldList",
    sp: "FieldList",
    A: "ArrayLike | None" = None,
    B: "ArrayLike | None" = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
) -> "FieldList": ...


def geopotential_on_hybrid_levels(
    t: "ArrayLike | FieldList",
    q: "ArrayLike | FieldList",
    zs: "ArrayLike | FieldList",
    sp: "ArrayLike | FieldList",
    A: "ArrayLike | None" = None,
    B: "ArrayLike | None" = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    vertical_dim: int = 0,
) -> "ArrayLike | FieldList":
    r"""Compute the geopotential on hybrid (IFS model) full-levels.

    Parameters
    ----------
    t : array-like | FieldList
        Temperature on hybrid full-levels (K). See the concrete implementations for details.
    q : array-like | FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same shape, level
        range and order as ``t``. See the concrete implementations for details.
    zs : array-like | FieldList
        Surface geopotential (m2/s2). See the concrete implementations for details.
    sp : array-like | FieldList
        Surface pressure (Pa). See the concrete implementations for details.
    A : array-like | None, optional
        A-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number.
        Can be None for FieldList/Field input if the coefficients can be resolved
        from metadata (FieldList only).
    B : array-like | None, optional
        B-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number.
        Must have the same size as ``A``.  Can be None for FieldList/Field input if
        the coefficients can be resolved from metadata (FieldList only).
    alpha_top : {"ifs", "arpege"}, optional
        Option to initialise alpha at the top of the model atmosphere. Default is ``"ifs"``.
        See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.
    vertical_dim : int, default=0
        Axis corresponding to the vertical coordinate in the input and output arrays.
        Default is 0. This keyword argument is not supported when the input data is FieldList.

    Returns
    -------
    array-like | FieldList
        Geopotential (m2/s2) on hybrid full-levels. See the concrete implementations for details.

    See Also
    --------
    pressure_on_hybrid_levels
    relative_geopotential_thickness_on_hybrid_levels


    Implementations
    ---------------
    :func:`geopotential_on_hybrid_levels` calls one of the following implementations depending
    on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.geopotential_on_hybrid_levels` for array-like
    - :py:meth:`earthkit.meteo.vertical.fieldlist.geopotential_on_hybrid_levels` for FieldList

    The function returns an object of the same type as the input arguments.
    """
    _kwargs = dict(A=A, B=B, alpha_top=alpha_top)
    if vertical_dim != 0:
        _kwargs["vertical_dim"] = vertical_dim

    return dispatch(geopotential_on_hybrid_levels, xarray=False, fieldlist=True, array=True)(t, q, zs, sp, **_kwargs)


@overload
def height_on_hybrid_levels(
    t: "ArrayLike",
    q: "ArrayLike",
    zs: "ArrayLike",
    sp: "ArrayLike",
    A: "ArrayLike",
    B: "ArrayLike",
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    h_type: Literal["geometric", "geopotential"] = "geometric",
    h_reference: Literal["ground", "sea"] = "ground",
    vertical_dim: int = 0,
) -> "ArrayLike": ...


@overload
def height_on_hybrid_levels(
    t: "FieldList",
    q: "FieldList",
    zs: "FieldList",
    sp: "FieldList",
    A: "ArrayLike | None" = ...,
    B: "ArrayLike | None" = ...,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    h_type: Literal["geometric", "geopotential"] = "geometric",
    h_reference: Literal["ground", "sea"] = "ground",
) -> "FieldList": ...


def height_on_hybrid_levels(
    t: "ArrayLike | FieldList",
    q: "ArrayLike | FieldList",
    zs: "ArrayLike | FieldList",
    sp: "ArrayLike | FieldList",
    A: "ArrayLike | None" = None,
    B: "ArrayLike | None" = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    h_type: Literal["geometric", "geopotential"] = "geometric",
    h_reference: Literal["ground", "sea"] = "ground",
    vertical_dim: int = 0,
) -> "ArrayLike | FieldList":
    r"""Compute the height on hybrid (IFS model) full-levels.

    Parameters
    ----------
    t : ArrayLike | FieldList
        Temperature on hybrid full-levels (K). See the concrete implementations for details.
    q : ArrayLike | FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same shape, level
        range and order as ``t``. See the concrete implementations for details.
    zs : ArrayLike | FieldList
        Surface geopotential (m2/s2). Not used when ``h_type`` is ``"geopotential"`` and
        ``h_reference`` is ``"ground"``. See the concrete implementations for details.
    sp : ArrayLike | FieldList
        Surface pressure (Pa). See the concrete implementations for details.
    A : array-like | None, optional
        A-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number.
        Can be None for FieldList/Field input if the coefficients can be resolved
        from metadata (FieldList only).
    B : array-like | None, optional
        B-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number.
        Must have the same size as ``A``. Can be None for FieldList/Field input if the
        coefficients can be resolved from metadata (FieldList only).
    alpha_top : {"ifs", "arpege"}, default="ifs"
        Option to initialise alpha at the top of the model atmosphere.
        See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.
    h_type : {"geometric", "geopotential"}, default="geometric"
        Type of height to compute. Possible values are:

        - ``"geometric"``: geometric height (m) with respect to ``h_reference``
        - ``"geopotential"``: geopotential height (m) with respect to ``h_reference``

    h_reference : {"ground", "sea"}, default="ground"
        Reference level for the height calculation. Possible values are:

        - ``"ground"``: height with respect to the ground/surface level
        - ``"sea"``: height with respect to sea level

    vertical_dim : int, default=0
        Axis corresponding to the vertical coordinate in the input and output arrays.
        Default is 0. This keyword argument is not supported when the input data is FieldList.

    Returns
    -------
        array-like | FieldList
            Height (m) on hybrid full-levels with respect to ``h_reference``. The type of height
            is defined by ``h_type``. See the concrete implementations for details.

    See Also
    --------
        hybrid_level_parameters
        pressure_on_hybrid_levels
        geopotential_on_hybrid_levels
        relative_geopotential_thickness_on_hybrid_levels


    Implementations
    ---------------
    :func:`height_on_hybrid_levels` calls one of the following implementations depending on
    the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.height_on_hybrid_levels` for array-like
    - :py:meth:`earthkit.meteo.vertical.fieldlist.height_on_hybrid_levels` for FieldList

    The function returns an object of the same type as the input arguments.
    """
    _kwargs = dict(A=A, B=B, alpha_top=alpha_top, h_type=h_type, h_reference=h_reference)
    if vertical_dim != 0:
        _kwargs["vertical_dim"] = vertical_dim

    return dispatch(height_on_hybrid_levels, xarray=False, fieldlist=True, array=True)(t, q, zs, sp, **_kwargs)


@overload
def interpolate_hybrid_to_pressure_levels(
    data: "ArrayLike",
    target_p: "ArrayLike",
    sp: "ArrayLike",
    A: "ArrayLike",
    B: "ArrayLike",
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    interpolation: Literal["linear", "log", "nearest"] = "linear",
    aux_bottom_data: "ArrayLike | None" = None,
    aux_bottom_p: "ArrayLike | None" = None,
    aux_top_data: "ArrayLike | None" = None,
    aux_top_p: "ArrayLike | None" = None,
    vertical_dim: int = 0,
) -> "ArrayLike": ...


@overload
def interpolate_hybrid_to_pressure_levels(
    data: "FieldList",
    target_p: "ArrayLike|FieldList|Field",
    sp: "FieldList|Field",
    A: "ArrayLike | None" = None,
    B: "ArrayLike | None" = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    interpolation: Literal["linear", "log", "nearest"] = "linear",
    aux_bottom_data: "float | FieldList | Field | None" = None,
    aux_bottom_p: "float | FieldList | Field | None" = None,
    aux_top_data: "float | FieldList | Field | None" = None,
    aux_top_p: "float | FieldList | Field | None" = None,
) -> "FieldList": ...


def interpolate_hybrid_to_pressure_levels(
    data: "ArrayLike | FieldList",
    target_p: "ArrayLike | FieldList | Field",
    sp: "ArrayLike | FieldList | Field",
    A: "ArrayLike | None" = None,
    B: "ArrayLike | None" = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    interpolation: Literal["linear", "log", "nearest"] = "linear",
    aux_bottom_data: "ArrayLike | float | FieldList | Field | None" = None,
    aux_bottom_p: "ArrayLike | float | FieldList | Field | None" = None,
    aux_top_data: "ArrayLike | float | FieldList | Field | None" = None,
    aux_top_p: "ArrayLike | float | FieldList | Field | None" = None,
    vertical_dim: int = 0,
) -> "ArrayLike | FieldList":
    r"""Interpolate data from hybrid full-levels (IFS model levels) to pressure levels.

    Parameters
    ----------
    data : ArrayLike | FieldList
        Data to be interpolated. Levels must be in ascending order with respect to the model
        level number. The axis corresponding to the vertical coordinate is defined by
        ``vertical_dim``.
    target_p : ArrayLike|FieldList|Field
        Target pressure (Pa). See the concrete implementations for details.
    sp : ArrayLike|FieldList|Field
        Surface pressure (Pa). See the concrete implementations for details.
    A : ArrayLike | None, optional
        A-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number.
        Can be None for FieldList/Field input if the coefficients can be resolved
        from metadata (FieldList only).
    B : ArrayLike | None, optional
        B-coefficients defining the hybrid levels. Must contain all the half-levels
        in ascending order with respect to the model level number.
        Must have the same size as ``A``. Can be None for FieldList/Field input if the
        coefficients can be resolved from metadata (FieldList only).
    alpha_top : {"ifs", "arpege"}, default="ifs"
        Option to initialise the alpha parameter at the top of the model atmosphere.
        See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.
    interpolation : {"linear", "log", "nearest"}, default="linear"
        Interpolation mode. Possible values:

        - ``"linear"``: linear interpolation in pressure
        - ``"log"``: linear interpolation in log-pressure
        - ``"nearest"``: nearest level interpolation

    aux_bottom_data : ArrayLike | float | FieldList | Field | None, optional
        Auxiliary data for interpolation to pressures between the bottom hybrid full-level
        and ``aux_bottom_p``. See the concrete implementations for details.
    aux_bottom_p : ArrayLike | float | FieldList | Field | None, optional
        Pressures (Pa) of ``aux_bottom_data``. See the concrete implementations for details.
    aux_top_data : ArrayLike | float | FieldList | Field | None, optional
        Auxiliary data for interpolation to pressures above the top hybrid full-level
        and below ``aux_top_p``. See the concrete implementations for details.
    aux_top_p : ArrayLike | float | FieldList | Field | None, optional
        Pressures (Pa) of ``aux_top_data``. See the concrete implementations for details.
    vertical_dim : int, default=0
        Axis corresponding to the vertical coordinate in the input and output arrays.
        Cannot be specified for FieldList/Field data. This keyword argument is not supported when
        the input data is FieldList.

    Returns
    -------
    array-like | FieldList
        Data interpolated to the target pressure. Values outside the available pressure
        range are set to nan. See the concrete implementations for details.


    See Also
    --------
    interpolate_monotonic


    Implementations
    ---------------
    :func:`interpolate_hybrid_to_pressure_levels` calls one of the following implementations
    depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.interpolate_hybrid_to_pressure_levels` for array-like
    - :py:meth:`earthkit.meteo.vertical.fieldlist.interpolate_hybrid_to_pressure_levels` for FieldList

    The function returns an object of the same type as the ``data`` argument.
    """
    _kawrgs = dict(
        alpha_top=alpha_top,
        interpolation=interpolation,
        aux_bottom_data=aux_bottom_data,
        aux_bottom_p=aux_bottom_p,
        aux_top_data=aux_top_data,
        aux_top_p=aux_top_p,
    )

    if vertical_dim != 0:
        _kawrgs["vertical_dim"] = vertical_dim

    return dispatch(interpolate_hybrid_to_pressure_levels, xarray=False, fieldlist=True, array=True)(
        data, target_p, sp, A, B, **_kawrgs
    )


@overload
def interpolate_hybrid_to_height_levels(
    data: "ArrayLike",
    target_h: "ArrayLike",
    t: "ArrayLike",
    q: "ArrayLike",
    za: "ArrayLike",
    sp: "ArrayLike",
    A: "ArrayLike",
    B: "ArrayLike",
    alpha_top: str = ...,
    h_type: str = ...,
    h_reference: str = ...,
    interpolation: str = ...,
    aux_bottom_data: "ArrayLike | None" = ...,
    aux_bottom_h: "ArrayLike | None" = ...,
    aux_top_data: "ArrayLike | None" = ...,
    aux_top_h: "ArrayLike | None" = ...,
    vertical_dim: int = ...,
) -> "ArrayLike": ...


@overload
def interpolate_hybrid_to_height_levels(
    data: "FieldList",
    target_h: "ArrayLike",
    t: "FieldList",
    q: "FieldList",
    za: "FieldList",
    sp: "FieldList",
    A: "ArrayLike | None" = ...,
    B: "ArrayLike | None" = ...,
    alpha_top: str = ...,
    h_type: str = ...,
    h_reference: str = ...,
    interpolation: str = ...,
    aux_bottom_data: "ArrayLike | None" = ...,
    aux_bottom_h: "ArrayLike | None" = ...,
    aux_top_data: "ArrayLike | None" = ...,
    aux_top_h: "ArrayLike | None" = ...,
) -> "FieldList": ...


def interpolate_hybrid_to_height_levels(
    data: "ArrayLike | FieldList",
    target_h: "ArrayLike",
    t: "ArrayLike | FieldList",
    q: "ArrayLike | FieldList",
    za: "ArrayLike | FieldList",
    sp: "ArrayLike | FieldList",
    A: "ArrayLike | None" = None,
    B: "ArrayLike | None" = None,
    alpha_top: str = "ifs",
    h_type: str = "geometric",
    h_reference: str = "ground",
    interpolation: str = "linear",
    aux_bottom_data: "ArrayLike | None" = None,
    aux_bottom_h: "ArrayLike | None" = None,
    aux_top_data: "ArrayLike | None" = None,
    aux_top_h: "ArrayLike | None" = None,
    vertical_dim: int = 0,
) -> "ArrayLike | FieldList":
    r"""Interpolate data from hybrid full-levels (IFS model levels) to height levels.

    Parameters
    ----------
    data : array-like | FieldList
        Data to be interpolated. Levels must be in ascending order with respect to the model
        level number. The axis corresponding to the vertical coordinate is defined by
        ``vertical_dim``.
    target_h : array-like
        Target height levels (m) to which ``data`` will be interpolated. Can be a scalar, a
        1D array, or a multidimensional array whose non-vertical axes match those of ``data``.
        The type of the height and the reference level are defined by ``h_type`` and
        ``h_reference``.
    t : array-like | FieldList
        Temperature on hybrid full-levels (K). Must have the same shape, level range and order
        as ``data``.
    q : array-like | FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same shape, level range
        and order as ``t``.
    za : array-like | FieldList
        Surface geopotential (m2/s2). Not used when ``h_type`` is ``"geopotential"`` and
        ``h_reference`` is ``"ground"``.
    sp : array-like | FieldList
        Surface pressure (Pa).
    A : array-like | None, optional
        A-coefficients defining the hybrid levels. If None (default), must be resolvable from
        metadata (FieldList only). See :func:`hybrid_level_parameters` for details.
    B : array-like | None, optional
        B-coefficients defining the hybrid levels. Must have the same size as ``A``. If None
        (default), must be resolvable from metadata (FieldList only).
    alpha_top : str, optional
        Option to initialise alpha at the top of the model atmosphere. Default is ``"ifs"``.
        See :func:`pressure_on_hybrid_levels` for details.
    h_type : str, optional
        Type of height. Default is ``"geometric"``. Possible values are ``"geometric"`` and
        ``"geopotential"``.
    h_reference : str, optional
        Reference level. Default is ``"ground"``. Possible values are ``"ground"`` and
        ``"sea"``.
    interpolation : str, optional
        Interpolation mode. Default is ``"linear"``. Possible values are ``"linear"``,
        ``"log"`` and ``"nearest"``.
    aux_bottom_data : array-like | None, optional
        Auxiliary data for interpolation to heights between the bottom hybrid full-level
        and ``aux_bottom_h``.
    aux_bottom_h : array-like | None, optional
        Heights (m) of ``aux_bottom_data``.
    aux_top_data : array-like | None, optional
        Auxiliary data for interpolation to heights above the top hybrid full-level and
        below ``aux_top_h``.
    aux_top_h : array-like | None, optional
        Heights (m) of ``aux_top_data``.
    vertical_dim : int, optional
        Axis corresponding to the vertical coordinate in the input and output arrays.
        Default is 0. This keyword argument is not supported when the input data is FieldList.

    Returns
    -------
    array-like | FieldList
        Data interpolated to the target height levels. Values outside the available height
        range are set to nan. The axis corresponding to the vertical coordinate is defined by
        ``vertical_dim``.


    See Also
    --------
    interpolate_monotonic


    Implementations
    ---------------
    :func:`interpolate_hybrid_to_height_levels` calls one of the following implementations
    depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.interpolate_hybrid_to_height_levels` for array-like
    - :py:meth:`earthkit.meteo.vertical.fieldlist.interpolate_hybrid_to_height_levels` for FieldList

    The function returns an object of the same type as the ``data`` argument.
    """
    _kwargs = dict(
        A=A,
        B=B,
        alpha_top=alpha_top,
        h_type=h_type,
        h_reference=h_reference,
        interpolation=interpolation,
        aux_bottom_data=aux_bottom_data,
        aux_bottom_h=aux_bottom_h,
        aux_top_data=aux_top_data,
        aux_top_h=aux_top_h,
    )
    if vertical_dim != 0:
        _kwargs["vertical_dim"] = vertical_dim
    return dispatch(interpolate_hybrid_to_height_levels, xarray=False, fieldlist=True, array=True)(
        data, target_h, t, q, za, sp, **_kwargs
    )


@overload
def interpolate_pressure_to_height_levels(
    data: "ArrayLike",
    target_h: "ArrayLike",
    z: "ArrayLike",
    zs: "ArrayLike",
    h_type: str = ...,
    h_reference: str = ...,
    interpolation: str = ...,
    aux_bottom_data: "ArrayLike | None" = ...,
    aux_bottom_h: "ArrayLike | None" = ...,
    aux_top_data: "ArrayLike | None" = ...,
    aux_top_h: "ArrayLike | None" = ...,
    vertical_dim: int = ...,
) -> "ArrayLike": ...


@overload
def interpolate_pressure_to_height_levels(
    data: "FieldList",
    target_h: "ArrayLike",
    z: "FieldList",
    zs: "FieldList",
    h_type: str = ...,
    h_reference: str = ...,
    interpolation: str = ...,
    aux_bottom_data: "ArrayLike | None" = ...,
    aux_bottom_h: "ArrayLike | None" = ...,
    aux_top_data: "ArrayLike | None" = ...,
    aux_top_h: "ArrayLike | None" = ...,
) -> "FieldList": ...


def interpolate_pressure_to_height_levels(
    data: "ArrayLike | FieldList",
    target_h: "ArrayLike",
    z: "ArrayLike | FieldList",
    zs: "ArrayLike | FieldList | None" = None,
    h_type: Literal["geometric", "geopotential"] = "geometric",
    h_reference: Literal["ground", "sea"] = "ground",
    interpolation: Literal["linear", "log", "nearest"] = "linear",
    aux_bottom_data: "ArrayLike | None" = None,
    aux_bottom_h: "ArrayLike | None" = None,
    aux_top_data: "ArrayLike | None" = None,
    aux_top_h: "ArrayLike | None" = None,
    vertical_dim: int = 0,
) -> "ArrayLike | FieldList":
    r"""Interpolate data from pressure levels to height levels.

    Parameters
    ----------
    data : array-like | FieldList
        Data to be interpolated. Levels must be monotonic (ascending or descending with
        respect to pressure). The axis corresponding to the vertical coordinate is defined
        by ``vertical_dim``.
    target_h : array-like
        Target height levels (m) to which ``data`` will be interpolated. Can be a scalar, a
        1D array, or a multidimensional array whose non-vertical axes match those of ``data``.
        The type of the height and the reference level are defined by ``h_type`` and
        ``h_reference``.
    z : array-like | FieldList
        Geopotential (m2/s2) on the same pressure levels as ``data``.
    zs : array-like | FieldList | None
        Surface geopotential (m2/s2). Only used when ``h_reference`` is ``"ground"``.
    h_type : str, optional
        Type of height. Default is ``"geometric"``. Possible values are ``"geometric"`` and
        ``"geopotential"``.
    h_reference : str, optional
        Reference level. Default is ``"ground"``. Possible values are ``"ground"`` and
        ``"sea"``.
    interpolation : str, optional
        Interpolation mode. Default is ``"linear"``. Possible values are ``"linear"``,
        ``"log"`` and ``"nearest"``.
    aux_bottom_data : array-like | None, optional
        Auxiliary data for interpolation to heights between the bottom pressure level and
        ``aux_bottom_h``.
    aux_bottom_h : array-like | None, optional
        Heights (m) of ``aux_bottom_data``.
    aux_top_data : array-like | None, optional
        Auxiliary data for interpolation to heights between the top pressure level and
        ``aux_top_h``.
    aux_top_h : array-like | None, optional
        Heights (m) of ``aux_top_data``.
    vertical_dim : int, optional
        Axis corresponding to the vertical coordinate in the input and output arrays.
        Default is 0. This keyword argument is not supported when the input data is FieldList.

    Returns
    -------
    array-like | FieldList
        Data interpolated to the target height levels. Values outside the available height
        range are set to nan. The axis corresponding to the vertical coordinate is defined by
        ``vertical_dim``.


    See Also
    --------
    interpolate_monotonic


    Implementations
    ---------------
    :func:`interpolate_pressure_to_height_levels` calls one of the following implementations
    depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.interpolate_pressure_to_height_levels` for array-like
    - :py:meth:`earthkit.meteo.vertical.fieldlist.interpolate_pressure_to_height_levels` for FieldList

    The function returns an object of the same type as the input arguments.
    """
    _kwargs = dict(
        h_type=h_type,
        h_reference=h_reference,
        interpolation=interpolation,
        aux_bottom_data=aux_bottom_data,
        aux_bottom_h=aux_bottom_h,
        aux_top_data=aux_top_data,
        aux_top_h=aux_top_h,
    )
    if vertical_dim != 0:
        _kwargs["vertical_dim"] = vertical_dim

    return dispatch(interpolate_pressure_to_height_levels, xarray=False, fieldlist=True, array=True)(
        data, target_h, z, zs, **_kwargs
    )


@overload
def interpolate_monotonic(
    data: "ArrayLike",
    coord: "ArrayLike",
    target_coord: "ArrayLike",
    interpolation: str = "linear",
    aux_min_level_data=None,
    aux_min_level_coord=None,
    aux_max_level_data=None,
    aux_max_level_coord=None,
    vertical_dim=None,
) -> "ArrayLike": ...


@overload
def interpolate_monotonic(
    data: "xarray.DataArray",
    coord: "xarray.DataArray",
    target_coord: "ArrayLike",
    coord_type: str | None = None,
    interpolation: str = "linear",
    vertical_dim=None,
) -> "xarray.DataArray": ...


@overload
def interpolate_monotonic(
    data: "FieldList",
    coord: "ArrayLike | FieldList",
    target_coord: "ArrayLike | FieldList",
    coord_type: str | None = None,
    interpolation: str = "linear",
    aux_min_level_data=None,
    aux_min_level_coord=None,
    aux_max_level_data=None,
    aux_max_level_coord=None,
) -> "FieldList": ...


def interpolate_monotonic(
    data: "ArrayLike | xarray.DataArray | FieldList",
    coord: "ArrayLike | xarray.DataArray | FieldList",
    target_coord: "ArrayLike | xarray.DataArray | FieldList",
    coord_type: str | None = None,
    interpolation: str = "linear",
    aux_min_level_data=None,
    aux_min_level_coord=None,
    aux_max_level_data=None,
    aux_max_level_coord=None,
    vertical_dim=None,
):
    r"""Interpolate data between the same type of monotonic coordinate levels.

    Parameters
    ----------
    data: array-like | xarray.DataArray | FieldList
        Data to be interpolated. Must have at least two fields/elements.
    coord: array-like | xarray.DataArray | FieldList
        Vertical coordinates related to ``data``.
    target_coord: array-like | xarray.DataArray | FieldList
        Target coordinate levels to which ``data`` will be interpolated.
    coord_type: str | None, optional
        Type of the coordinate levels. This keyword argument is not supported
        when the input data is ArrayLike.
    interpolation: str, optional
        Interpolation mode ("linear", "log", or "nearest").
    aux_min_level_data: optional
        Auxiliary data for minimum level extrapolation. This keyword argument is not supported
        when the input data is Xarray..
    aux_min_level_coord: optional
        Coordinates of auxiliary minimum level data. This keyword argument is not supported
        when the input data is Xarray.
    aux_max_level_data: optional
        Auxiliary data for maximum level extrapolation. This keyword argument is not supported
        when the input data is Xarray.
    aux_max_level_coord: optional
        Coordinates of auxiliary maximum level data. This keyword argument is not supported
        when the input data is Xarray.
    vertical_dim: int | str | None, optional
        Vertical dimension specification. This keyword argument is not supported when the input
        data is FieldList.

    Returns
    -------
    array-like | xarray.DataArray | FieldList
        Interpolated data at target coordinate levels.


    Implementations
    ---------------
    :func:`interpolate_monotonic` calls one of the following implementations
    depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.array.interpolate_monotonic` for array-like
    - :py:meth:`earthkit.meteo.vertical.xarray.interpolate_monotonic` for xarray.DataArray
    - :py:meth:`earthkit.meteo.vertical.fieldlist.interpolate_monotonic` for FieldList

    The function returns an object of the same type as the ``data`` argument.

    """
    _kwargs = dict(
        interpolation=interpolation,
        aux_min_level_data=aux_min_level_data,
        aux_min_level_coord=aux_min_level_coord,
        aux_max_level_data=aux_max_level_data,
        aux_max_level_coord=aux_max_level_coord,
    )

    if vertical_dim is not None:
        _kwargs["vertical_dim"] = vertical_dim
    if coord_type is not None:
        _kwargs["coord_type"] = coord_type

    return dispatch(interpolate_monotonic, xarray=True, fieldlist=True, array=True)(
        data, coord, target_coord, **_kwargs
    )


def interpolate_to_pressure_levels(data, p, target_p, target_p_units="Pa", interpolation="linear", vertical_dim="z"):
    r"""Interpolate data to pressure levels.

    Parameters
    ----------
    data: xarray.DataArray
        Data to be interpolated.
    p:  xarray.DataArray
        Pressure coordinate of ``data``.
    target_p: ArrayLike
        Target pressure levels.
    target_p_units: str, optional
        Units of ``target_p`` (default: "Pa").
    interpolation: str, optional
        Interpolation mode (default: "linear").
    vertical_dim: str, optional
        Vertical dimension (default: "z").

    Returns
    -------
    xarray.DataArray
        Interpolated data at target pressure levels.


    Implementations
    ---------------
    :func:`interpolate_to_pressure_levels` calls one of the following implementations
    depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.xarray.interpolate_to_pressure_levels` for xarray.DataArray

    The function returns an object of the same type as the ``data`` argument.

    """
    return dispatch(interpolate_to_pressure_levels, xarray=True, fieldlist=False, array=False)(
        data,
        p,
        target_p,
        target_p_units=target_p_units,
        interpolation=interpolation,
        vertical_dim=vertical_dim,
    )


def interpolate_sleve_to_coord_levels(data, h, coord, target_coord, folding_mode="undef_fold", vertical_dim="z"):
    r"""Interpolate data from SLEVE levels to coordinate levels.

    Parameters
    ----------
    data: xarray.DataArray
        Data on SLEVE levels.
    h: xarray.DataArray
        SLEVE coordinate values.
    coord: xarray.DataArray
        Reference coordinate values.
    target_coord: ArrayLike
        Target coordinate levels.
    folding_mode: str, optional
        Folding mode (default: "undef_fold").
    vertical_dim: str, optional
        Vertical dimension (default: "z").

    Returns
    -------
    xarray.DataArray
        Interpolated data at target coordinate levels.


    Implementations
    ---------------
    :func:`interpolate_sleve_to_coord_levels` calls one of the following implementations
    depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.xarray.interpolate_sleve_to_coord_levels` for xarray.DataArray

    The function returns an object of the same type as the ``data`` argument.

    """
    return dispatch(interpolate_sleve_to_coord_levels, xarray=True, fieldlist=False, array=False)(
        data, h, coord, target_coord, folding_mode=folding_mode, vertical_dim=vertical_dim
    )


def interpolate_sleve_to_theta_levels(
    data, g, theta, target_theta, target_t_units="K", folding_mode="undef_fold", vertical_dim="z"
):
    r"""Interpolate data from SLEVE levels to theta (potential temperature) levels.

    Parameters
    ----------
    data: xarray.DataArray
        Data on SLEVE levels.
    g: xarray.DataArray
        Gravity or related SLEVE parameter.
    theta: xarray.DataArray
        Reference theta values.
    target_theta: ArrayLike
        Target theta levels.
    target_t_units: str, optional
        Units of target theta (default: "K").
    folding_mode: str, optional
        Folding mode (default: "undef_fold").
    vertical_dim: str, optional
        Vertical dimension (default: "z").

    Returns
    -------
    xarray.DataArray
        Interpolated data at target theta levels.


    Implementations
    ---------------
    :func:`interpolate_sleve_to_theta_levels` calls one of the following implementations
    depending on the type of the input arguments:

    - :py:meth:`earthkit.meteo.vertical.xarray.interpolate_sleve_to_theta_levels` for xarray.DataArray

    The function returns an object of the same type as the ``data`` argument.

    """
    return dispatch(interpolate_sleve_to_theta_levels, xarray=True, fieldlist=False, array=False)(
        data,
        g,
        theta,
        target_theta,
        target_t_units=target_t_units,
        folding_mode=folding_mode,
        vertical_dim=vertical_dim,
    )
