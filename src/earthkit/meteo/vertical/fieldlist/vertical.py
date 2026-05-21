# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from earthkit.data import Field, FieldList  # type: ignore[import]
from numpy.typing import ArrayLike

from earthkit.meteo import constants
from earthkit.meteo.utils.decorators import fieldlist_ufunc

from .. import array


def geopotential_height_from_geopotential(
    z: FieldList | Field,
) -> FieldList | Field:
    r"""Compute geopotential height from geopotential.

    Parameters
    ----------
    z: FieldList|Field
        Geopotential (m2/s2).

    Returns
    -------
    FieldList|Field
        Geopotential height (m). The result has the same type as the input
        (FieldList or Field).

    Notes
    -----
    The computation is based on the following definition:

    .. math::

        gh = \frac{z}{g}

    where :math:`g` is the gravitational acceleration on the surface of
    the Earth (see :py:attr:`meteo.constants.g`).
    """
    fieldlist_ufunc_kwargs = {"default": "gh", "param_unit": "gpm"}

    return fieldlist_ufunc(
        array.geopotential_height_from_geopotential, z, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def geopotential_from_geopotential_height(
    gh: FieldList | Field,
) -> FieldList | Field:
    r"""Compute geopotential from geopotential height.

    Parameters
    ----------
    gh: FieldList|Field
        Geopotential height (m).

    Returns
    -------
    FieldList|Field
        Geopotential (m2/s2). The result has the same type as the input
        (FieldList or Field).

    Notes
    -----
    The computation is based on the following definition:

    .. math::

        z = gh \cdot g

    where :math:`g` is the gravitational acceleration on the surface of
    the Earth (see :py:attr:`meteo.constants.g`).
    """
    fieldlist_ufunc_kwargs = {"default": "z", "param_unit": "m2/s2"}

    return fieldlist_ufunc(
        array.geopotential_from_geopotential_height, gh, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def geopotential_height_from_geometric_height(
    h: FieldList | Field,
    R_earth: float = constants.R_earth,
) -> FieldList | Field:
    r"""Compute geopotential height from geometric height.

    Parameters
    ----------
    h: FieldList|Field
        Geometric height with respect to the sea level (m).
    R_earth: float, optional
        Average radius of the Earth (m).

    Returns
    -------
    FieldList|Field
        Geopotential height (m). The result has the same type as the input
        (FieldList or Field).

    Notes
    -----
    The computation is based on the following formula:

    .. math::

        gh = \frac{h \cdot R_{earth}}{R_{earth} + h}

    where :math:`R_{earth}` is the average radius of the Earth
    (see :py:attr:`meteo.constants.R_earth`).
    """
    fieldlist_ufunc_kwargs = {"default": "gh", "param_unit": "gpm"}

    return fieldlist_ufunc(
        array.geopotential_height_from_geometric_height,
        h,
        R_earth,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
    )


def geopotential_from_geometric_height(
    h: FieldList | Field,
    R_earth: float = constants.R_earth,
) -> FieldList | Field:
    r"""Compute geopotential from geometric height.

    Parameters
    ----------
    h: FieldList|Field
        Geometric height with respect to the sea level (m).
    R_earth: float, optional
        Average radius of the Earth (m).

    Returns
    -------
    FieldList|Field
        Geopotential (m2/s2). The result has the same type as the input
        (FieldList or Field).

    Notes
    -----
    The computation is based on the following formula:

    .. math::

        z = \frac{h \cdot g \cdot R_{earth}}{R_{earth} + h}

    where

        * :math:`R_{earth}` is the average radius of the Earth
          (see :py:attr:`meteo.constants.R_earth`)
        * :math:`g` is the gravitational acceleration on the surface of
          the Earth (see :py:attr:`meteo.constants.g`)
    """
    fieldlist_ufunc_kwargs = {"default": "z", "param_unit": "m2/s2"}

    return fieldlist_ufunc(
        array.geopotential_from_geometric_height, h, R_earth, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def geometric_height_from_geopotential_height(
    gh: FieldList | Field,
    R_earth: float = constants.R_earth,
) -> FieldList | Field:
    r"""Compute geometric height from geopotential height.

    Parameters
    ----------
    gh: FieldList|Field
        Geopotential height (m).
    R_earth: float, optional
        Average radius of the Earth (m).

    Returns
    -------
    FieldList|Field
        Geometric height (m). The result has the same type as the input
        (FieldList or Field).

    Notes
    -----
    The computation is based on the following formula:

    .. math::

        h = \frac{R_{earth} \cdot gh}{R_{earth} - gh}

    where :math:`R_{earth}` is the average radius of the Earth
    (see :py:attr:`meteo.constants.R_earth`).
    """
    fieldlist_ufunc_kwargs = {"default": "h", "param_unit": "m"}

    return fieldlist_ufunc(
        array.geometric_height_from_geopotential_height,
        gh,
        R_earth,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
    )


def geometric_height_from_geopotential(
    z: FieldList | Field,
    R_earth: float = constants.R_earth,
) -> FieldList | Field:
    r"""Compute geometric height from geopotential.

    Parameters
    ----------
    z: FieldList|Field
        Geopotential (m2/s2).
    R_earth: float, optional
        Average radius of the Earth (m).

    Returns
    -------
    FieldList|Field
        Geometric height (m). The result has the same type as the input
        (FieldList or Field).

    Notes
    -----
    The computation is based on the following formula:

    .. math::

        h = \frac{R_{earth} \cdot \frac{z}{g}}{R_{earth} - \frac{z}{g}}

    where

        * :math:`R_{earth}` is the average radius of the Earth
          (see :py:attr:`meteo.constants.R_earth`)
        * :math:`g` is the gravitational acceleration on the surface of
          the Earth (see :py:attr:`meteo.constants.g`)
    """
    fieldlist_ufunc_kwargs = {"default": "h", "param_unit": "m"}

    return fieldlist_ufunc(
        array.geometric_height_from_geopotential, z, R_earth, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )


def pressure_on_hybrid_levels(
    sp: FieldList | Field,
    levels: ArrayLike | list | tuple | None = None,
    A: ArrayLike | None = None,
    B: ArrayLike | None = None,
    alpha_top: str = "ifs",
    output: str | list | tuple = "full",
) -> FieldList | tuple[FieldList, ...]:
    r"""Compute pressure and related parameters on hybrid (IFS model) levels.

    Parameters
    ----------
    sp: FieldList|Field
        Surface pressure (Pa).
    levels: ArrayLike | list | tuple | None, optional
        Hybrid full-levels to return. Level numbering starts at 1 at the top
        of the atmosphere and increases towards the surface. If None (default),
        all levels are returned.
        number of fields and level ordering as ``t``.
    A: ArrayLike | None
        A-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number
        (from the top of the atmosphere toward the surface).
    B: ArrayLike | None
        B-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        (from the top of the atmosphere toward the surface).
        Must have the same size as ``A``.
    alpha_top: str
        Option to initialise the alpha parameter on the top of the model
        atmosphere. See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.
    output: str|list|tuple[str]|None
        Which outputs to return. Possible values are ``"full"``, ``"half"``,
        ``"delta"`` and ``"alpha"``. Default is ``"full"``.

    Returns
    -------
    FieldList|tuple[FieldList, ...]
        Pressure and/or related parameters on hybrid levels. When a single
        output type is requested, a single FieldList is returned. When
        multiple output types are requested, a tuple of FieldLists is
        returned.

    See Also
    --------
    earthkit.meteo.vertical.array.pressure_on_hybrid_levels
    """
    fieldlist_ufunc_kwargs = {"default": "pres", "param_unit": "Pa"}

    if A is not None and B is None:
        raise ValueError("When A is provided, B must also be provided.")
    if A is None and B is not None:
        raise ValueError("When B is provided, A must also be provided.")
    if A is None or B is None:
        from ..array.hybrid import _hybrid_level_parameters_from_fieldlist

        A, B = _hybrid_level_parameters_from_fieldlist(sp, levels)
        if A is None or B is None:
            raise ValueError("A and B parameters could not be inferred from the input fields.")

    assert A is not None and B is not None, "A and B parameters must be provided or inferred from the input fields."

    if len(A) != len(B):
        raise ValueError("A and B must have the same length.")

    return fieldlist_ufunc(
        array.pressure_on_hybrid_levels,
        sp,
        levels,
        A=A,
        B=B,
        alpha_top=alpha_top,
        output=output,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
    )


def relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta(
    t: FieldList,
    q: FieldList,
    alpha: FieldList,
    delta: FieldList,
) -> FieldList:
    r"""Compute the geopotential thickness between the surface and hybrid full-levels from alpha and delta.

    Parameters
    ----------
    t: FieldList
        Temperature on hybrid full-levels (K). Each field corresponds to one
        model level.
    q: FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same
        number of fields and level ordering as ``t``.
    alpha: FieldList
        Alpha parameter computed using
        :func:`pressure_on_hybrid_levels`. Must have the same number of
        fields and level ordering as ``t``.
    delta: FieldList
        Delta parameter computed using
        :func:`pressure_on_hybrid_levels`. Must have the same number of
        fields and level ordering as ``t``.

    Returns
    -------
    FieldList
        Geopotential thickness (m2/s2) between the surface and hybrid
        full-levels.

    See Also
    --------
    pressure_on_hybrid_levels
    earthkit.meteo.vertical.array.relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta
    """
    pass


def relative_geopotential_thickness_on_hybrid_levels(
    t: FieldList,
    q: FieldList,
    A: ArrayLike,
    B: ArrayLike,
    sp: FieldList | Field,
    alpha_top: str = "ifs",
) -> FieldList:
    r"""Compute the geopotential thickness between the surface and hybrid full-levels.

    Parameters
    ----------
    t: FieldList
        Temperature on hybrid full-levels (K). Each field corresponds to one
        model level. Levels must be in ascending order with respect to the model
        level number.
    q: FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same
        number of fields and level ordering as ``t``.
    A: ArrayLike
        A-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
    B: ArrayLike
        B-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        Must have the same size as ``A``.
    sp: FieldList|Field
        Surface pressure (Pa).
    alpha_top: str
        Option to initialise the alpha parameter on the top of the model
        atmosphere. See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.

    Returns
    -------
    FieldList
        Geopotential thickness (m2/s2) between the surface and hybrid
        full-levels.

    See Also
    --------
    pressure_on_hybrid_levels
    relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta
    earthkit.meteo.vertical.array.relative_geopotential_thickness_on_hybrid_levels
    """
    pass


def geopotential_on_hybrid_levels(
    t: FieldList,
    q: FieldList,
    zs: FieldList | Field,
    A: ArrayLike,
    B: ArrayLike,
    sp: FieldList | Field,
    alpha_top: str = "ifs",
) -> FieldList:
    r"""Compute geopotential on hybrid (IFS model) full-levels.

    Parameters
    ----------
    t: FieldList
        Temperature on hybrid full-levels (K). Each field corresponds to one
        model level. Levels must be in ascending order with respect to the model
        level number.
    q: FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same
        number of fields and level ordering as ``t``.
    zs: FieldList|Field
        Surface geopotential (m2/s2).
    A: ArrayLike
        A-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
    B: ArrayLike
        B-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        Must have the same size as ``A``.
    sp: FieldList|Field
        Surface pressure (Pa).
    alpha_top: str
        Option to initialise the alpha parameter on the top of the model
        atmosphere. See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.

    Returns
    -------
    FieldList
        Geopotential (m2/s2) on hybrid full-levels.

    See Also
    --------
    pressure_on_hybrid_levels
    relative_geopotential_thickness_on_hybrid_levels
    earthkit.meteo.vertical.array.geopotential_on_hybrid_levels
    """
    pass


def height_on_hybrid_levels(
    t: FieldList,
    q: FieldList,
    zs: FieldList | Field,
    A: ArrayLike,
    B: ArrayLike,
    sp: FieldList | Field,
    alpha_top: str = "ifs",
    h_type: str = "geometric",
    h_reference: str = "ground",
) -> FieldList:
    r"""Compute the height on hybrid (IFS model) full-levels.

    Parameters
    ----------
    t: FieldList
        Temperature on hybrid full-levels (K). Each field corresponds to one
        model level. Levels must be in ascending order with respect to the model
        level number.
    q: FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same
        number of fields and level ordering as ``t``.
    zs: FieldList|Field
        Surface geopotential (m2/s2). Not used when ``h_type`` is
        ``"geopotential"`` and ``h_reference`` is ``"ground"``.
    A: ArrayLike
        A-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
    B: ArrayLike
        B-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        Must have the same size as ``A``.
    sp: FieldList|Field
        Surface pressure (Pa).
    alpha_top: str
        Option to initialise the alpha parameter on the top of the model
        atmosphere. See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.
    h_type: str
        Type of height to compute. Default is ``"geometric"``. Possible values:

        - ``"geometric"``: geometric height (m)
        - ``"geopotential"``: geopotential height (m)

    h_reference: str
        Reference level for the height calculation. Default is ``"ground"``.
        Possible values:

        - ``"ground"``: height with respect to the ground/surface level
        - ``"sea"``: height with respect to the sea level

    Returns
    -------
    FieldList
        Height (m) on hybrid full-levels.

    See Also
    --------
    geopotential_on_hybrid_levels
    relative_geopotential_thickness_on_hybrid_levels
    earthkit.meteo.vertical.array.height_on_hybrid_levels
    """
    pass


def interpolate_hybrid_to_pressure_levels(
    data: FieldList,
    target_p: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    sp: FieldList | Field,
    alpha_top: str = "ifs",
    interpolation: str = "linear",
    aux_bottom_data: FieldList | Field | None = None,
    aux_bottom_p: ArrayLike | None = None,
    aux_top_data: FieldList | Field | None = None,
    aux_top_p: ArrayLike | None = None,
) -> FieldList:
    r"""Interpolate data from hybrid full-levels to pressure levels.

    Parameters
    ----------
    data: FieldList
        Data to be interpolated. Each field corresponds to one hybrid
        full-level. Levels must be in ascending order with respect to the model
        level number.
    target_p: ArrayLike
        Target pressure levels (Pa).
    A: ArrayLike
        A-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
    B: ArrayLike
        B-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        Must have the same size as ``A``.
    sp: FieldList|Field
        Surface pressure (Pa).
    alpha_top: str
        Option to initialise the alpha parameter on the top of the model
        atmosphere. See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.
    interpolation: str
        Interpolation mode. Default is ``"linear"``. Possible values:

        - ``"linear"``: linear interpolation in pressure
        - ``"log"``: linear interpolation in log-pressure
        - ``"nearest"``: nearest level interpolation

    aux_bottom_data: FieldList|Field|None, optional
        Auxiliary data for interpolation below the bottom hybrid full-level.
    aux_bottom_p: ArrayLike|None, optional
        Pressures (Pa) of ``aux_bottom_data``.
    aux_top_data: FieldList|Field|None, optional
        Auxiliary data for interpolation above the top hybrid full-level.
    aux_top_p: ArrayLike|None, optional
        Pressures (Pa) of ``aux_top_data``.

    Returns
    -------
    FieldList
        Data interpolated to the target pressure levels. When interpolation is
        not possible for a given target pressure level, the corresponding output
        values are set to NaN.

    See Also
    --------
    interpolate_monotonic
    earthkit.meteo.vertical.array.interpolate_hybrid_to_pressure_levels
    """
    pass


def interpolate_hybrid_to_height_levels(
    data: FieldList,
    target_h: ArrayLike,
    t: FieldList,
    q: FieldList,
    zs: FieldList | Field,
    A: ArrayLike,
    B: ArrayLike,
    sp: FieldList | Field,
    alpha_top: str = "ifs",
    h_type: str = "geometric",
    h_reference: str = "ground",
    interpolation: str = "linear",
    aux_bottom_data: FieldList | Field | None = None,
    aux_bottom_h: ArrayLike | None = None,
    aux_top_data: FieldList | Field | None = None,
    aux_top_h: ArrayLike | None = None,
) -> FieldList:
    r"""Interpolate data from hybrid full-levels to height levels.

    Parameters
    ----------
    data: FieldList
        Data to be interpolated. Each field corresponds to one hybrid
        full-level. Levels must be in ascending order with respect to the model
        level number.
    target_h: ArrayLike
        Target height levels (m). The type and reference of the height are
        defined by ``h_type`` and ``h_reference``.
    t: FieldList
        Temperature on hybrid full-levels (K). Must have the same number of
        fields and level ordering as ``data``.
    q: FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same
        number of fields and level ordering as ``data``.
    zs: FieldList|Field
        Surface geopotential (m2/s2).
    A: ArrayLike
        A-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
    B: ArrayLike
        B-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        Must have the same size as ``A``.
    sp: FieldList|Field
        Surface pressure (Pa).
    alpha_top: str
        Option to initialise the alpha parameter on the top of the model
        atmosphere. See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.
    h_type: str
        Type of height to compute. Default is ``"geometric"``. Possible values:

        - ``"geometric"``: geometric height (m)
        - ``"geopotential"``: geopotential height (m)

    h_reference: str
        Reference level for the height calculation. Default is ``"ground"``.
        Possible values:

        - ``"ground"``: height with respect to the ground/surface level
        - ``"sea"``: height with respect to the sea level

    interpolation: str
        Interpolation mode. Default is ``"linear"``. Possible values:

        - ``"linear"``: linear interpolation in height
        - ``"log"``: linear interpolation in log-height
        - ``"nearest"``: nearest level interpolation

    aux_bottom_data: FieldList|Field|None, optional
        Auxiliary data for interpolation below the bottom hybrid full-level.
    aux_bottom_h: ArrayLike|None, optional
        Heights (m) of ``aux_bottom_data``.
    aux_top_data: FieldList|Field|None, optional
        Auxiliary data for interpolation above the top hybrid full-level.
    aux_top_h: ArrayLike|None, optional
        Heights (m) of ``aux_top_data``.

    Returns
    -------
    FieldList
        Data interpolated to the target height levels. When interpolation is
        not possible for a given target height level, the corresponding output
        values are set to NaN.

    See Also
    --------
    interpolate_monotonic
    height_on_hybrid_levels
    earthkit.meteo.vertical.array.interpolate_hybrid_to_height_levels
    """
    pass


def interpolate_pressure_to_height_levels(
    data: FieldList,
    target_h: ArrayLike,
    z: FieldList,
    zs: FieldList | Field,
    h_type: str = "geometric",
    h_reference: str = "ground",
    interpolation: str = "linear",
    aux_bottom_data: FieldList | Field | None = None,
    aux_bottom_h: ArrayLike | None = None,
    aux_top_data: FieldList | Field | None = None,
    aux_top_h: ArrayLike | None = None,
) -> FieldList:
    r"""Interpolate data from pressure levels to height levels.

    Parameters
    ----------
    data: FieldList
        Data to be interpolated. Each field corresponds to one pressure level.
        Levels must be monotonically ordered with respect to pressure.
    target_h: ArrayLike
        Target height levels (m). The type and reference of the height are
        defined by ``h_type`` and ``h_reference``.
    z: FieldList
        Geopotential (m2/s2) on the same pressure levels as ``data``.
    zs: FieldList|Field
        Surface geopotential (m2/s2). Only used when ``h_reference`` is
        ``"ground"``.
    h_type: str
        Type of height to compute. Default is ``"geometric"``. Possible values:

        - ``"geometric"``: geometric height (m)
        - ``"geopotential"``: geopotential height (m)

    h_reference: str
        Reference level for the height calculation. Default is ``"ground"``.
        Possible values:

        - ``"ground"``: height with respect to the ground/surface level
        - ``"sea"``: height with respect to the sea level

    interpolation: str
        Interpolation mode. Default is ``"linear"``. Possible values:

        - ``"linear"``: linear interpolation in height
        - ``"log"``: linear interpolation in log-height
        - ``"nearest"``: nearest level interpolation

    aux_bottom_data: FieldList|Field|None, optional
        Auxiliary data for interpolation below the bottom pressure level.
    aux_bottom_h: ArrayLike|None, optional
        Heights (m) of ``aux_bottom_data``.
    aux_top_data: FieldList|Field|None, optional
        Auxiliary data for interpolation above the top pressure level.
    aux_top_h: ArrayLike|None, optional
        Heights (m) of ``aux_top_data``.

    Returns
    -------
    FieldList
        Data interpolated to the target height levels. When interpolation is
        not possible for a given target height level, the corresponding output
        values are set to NaN.

    See Also
    --------
    interpolate_monotonic
    earthkit.meteo.vertical.array.interpolate_pressure_to_height_levels
    """
    pass


def interpolate_monotonic(
    data: FieldList,
    coord: FieldList,
    target_coord: ArrayLike,
    interpolation: str = "linear",
    aux_min_level_data: FieldList | Field | None = None,
    aux_min_level_coord: ArrayLike | None = None,
    aux_max_level_data: FieldList | Field | None = None,
    aux_max_level_coord: ArrayLike | None = None,
) -> FieldList:
    r"""Interpolate data between the same type of monotonic coordinate levels.

    Parameters
    ----------
    data: FieldList
        Data to be interpolated. Each field corresponds to one level.
        Must have at least two fields.
    coord: FieldList
        Vertical coordinates related to ``data``. Must have the same number
        of fields as ``data``. Must be monotonic along the vertical axis.
    target_coord: ArrayLike
        Target coordinate levels to which ``data`` will be interpolated.
    interpolation: str
        Interpolation mode. Default is ``"linear"``. Possible values:

        - ``"linear"``: linear interpolation between the two nearest levels
        - ``"log"``: linear interpolation in logarithm of coordinate
        - ``"nearest"``: nearest level interpolation

    aux_min_level_data: FieldList|Field|None, optional
        Auxiliary data for interpolation to target levels below the minimum
        coordinate level.
    aux_min_level_coord: ArrayLike|None, optional
        Coordinates of ``aux_min_level_data``.
    aux_max_level_data: FieldList|Field|None, optional
        Auxiliary data for interpolation to target levels above the maximum
        coordinate level.
    aux_max_level_coord: ArrayLike|None, optional
        Coordinates of ``aux_max_level_data``.

    Returns
    -------
    FieldList
        Data interpolated to the target levels. When interpolation is not
        possible for a given target level, the corresponding output values
        are set to NaN.

    See Also
    --------
    earthkit.meteo.vertical.array.interpolate_monotonic
    """
    pass
