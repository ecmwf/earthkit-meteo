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
from earthkit.meteo.utils.fieldlist import get_hybrid_level_parameters, surface_pressure_values
from earthkit.meteo.utils.param import FIELD_PARAMS

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
        R_earth=R_earth,
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
        array.geopotential_from_geometric_height,
        h,
        R_earth=R_earth,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
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
        R_earth=R_earth,
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
        array.geometric_height_from_geopotential,
        z,
        R_earth=R_earth,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
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
    output : str|list|tuple
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
          :func:`relative_geopotential_thickness_on_hybrid_levels` for details..

    Returns
    -------
    FieldList|tuple[FieldList, ...]
        Pressure and/or related parameters on hybrid levels. When a single
        output type is requested, a single FieldList is returned. When
        multiple output types are requested, a tuple of FieldLists is
        returned, one for each requested output type, in the same order
        as specified in the input.

    See Also
    --------
    earthkit.meteo.vertical.array.pressure_on_hybrid_levels
    """
    if isinstance(sp, FieldList):
        if len(sp) != 1:
            raise ValueError(f"Expected exactly one surface pressure field, but found {len(sp)}.")
        sp = sp[0]
    if not isinstance(sp, Field):
        raise ValueError("Surface pressure must be a Field or a FieldList containing exactly one Field.")

    A, B = get_hybrid_level_parameters(sp, A=A, B=B)

    if isinstance(output, str):
        output = [
            output,
        ]
    else:
        output = list(output)

    if "level" in output:
        raise ValueError("Output type 'level' is not supported for the fieldlist version of pressure_on_hybrid_levels.")

    output.append("level")
    output = tuple(output)

    params = {
        "full": FIELD_PARAMS.get("pressure_full_level"),
        "half": FIELD_PARAMS.get("pressure_half_level"),
        "delta": FIELD_PARAMS.get("hybrid_delta"),
        "alpha": FIELD_PARAMS.get("hybrid_alpha"),
    }

    def _output(template, fl_values, fl_levels, name):
        for rv, rl in zip(fl_values, fl_levels):
            yield template.set(
                values=rv,
                parameter=params[name],
                vertical={"level": rl, "level_type": "hybrid"},
            )

    res_keys = output[:-1]  # all output types except "level"
    results = [[] for _ in res_keys]

    sp = [sp] if isinstance(sp, Field) else sp

    for field in sp:
        sp_values = surface_pressure_values(field)
        res_values = array.pressure_on_hybrid_levels(
            sp_values,
            levels=levels,
            A=A,
            B=B,
            alpha_top=alpha_top,
            output=output,
        )

        if len(results) == 1:
            res_values = [res_values]

        res_levels = res_values[-1]
        res_values = res_values[:-1]

        for i in range(len(res_keys)):
            if res_keys[i] in ["full", "delta", "alpha"]:
                fl_levels = res_levels["full"]
            elif res_keys[i] == "half":
                fl_levels = res_levels["half"]
            results[i].extend(_output(field, res_values[i], fl_levels, res_keys[i]))

    if len(results) == 1:
        return FieldList.from_fields(results[0])
    else:
        return tuple(FieldList.from_fields(r) for r in results)


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
    from .hybrid import _HybridInput

    _hybrid = _HybridInput()
    _hybrid.add_t(t)
    _hybrid.add_q(q)
    _hybrid.add_alpha(alpha)
    _hybrid.add_delta(delta)
    _hybrid.check_levels()  # check that all input FieldLists have the same levels and return the levels

    t_arr = _hybrid.t.to_numpy(copy=False)
    q_arr = _hybrid.q.to_numpy(copy=False)
    alpha_arr = _hybrid.alpha.to_numpy(copy=False)
    delta_arr = _hybrid.delta.to_numpy(copy=False)

    res_arr = array.relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta(
        t=t_arr,
        q=q_arr,
        alpha=alpha_arr,
        delta=delta_arr,
    )

    return _hybrid.to_fieldlist(res_arr, template=t[0], param_name="relative_geopotential_thickness")


def relative_geopotential_thickness_on_hybrid_levels(
    t: FieldList,
    q: FieldList,
    sp: FieldList | Field,
    A: ArrayLike | None = None,
    B: ArrayLike | None = None,
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
    sp: FieldList|Field
        Surface pressure (Pa).
    A: ArrayLike
        A-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
    B: ArrayLike
        B-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        Must have the same size as ``A``.
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
    from .hybrid import _HybridInput

    _hybrid = _HybridInput()
    _hybrid.add_t(t)
    _hybrid.add_q(q)
    _hybrid.add_sp(sp)
    _hybrid.generate_AB(A, B)
    _hybrid.check_levels()  # check that all input FieldLists have the same levels and return the levels

    t_arr = _hybrid.t.to_numpy(copy=False)
    q_arr = _hybrid.q.to_numpy(copy=False)
    sp_arr = _hybrid.sp.to_numpy(copy=False)
    A = _hybrid.A
    B = _hybrid.B

    res_arr = array.relative_geopotential_thickness_on_hybrid_levels(
        t=t_arr,
        q=q_arr,
        sp=sp_arr,
        A=A,
        B=B,
        alpha_top=alpha_top,
    )

    return _hybrid.to_fieldlist(res_arr, template=t[0], param_name="relative_geopotential_thickness")


def geopotential_on_hybrid_levels(
    t: FieldList,
    q: FieldList,
    zs: FieldList | Field,
    sp: FieldList | Field,
    A: ArrayLike | None = None,
    B: ArrayLike | None = None,
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
    from .hybrid import _HybridInput

    _hybrid = _HybridInput()
    _hybrid.add_sp(sp)
    _hybrid.add_zs(zs)
    _hybrid.add_t(t)
    _hybrid.add_q(q)
    _hybrid.generate_AB(A, B)
    _hybrid.check_levels()  # check that all input FieldLists have the same levels and return the levels

    t_arr = _hybrid.t.to_numpy(copy=False)
    q_arr = _hybrid.q.to_numpy(copy=False)
    zs_arr = _hybrid.zs.to_numpy(copy=False)
    sp_arr = _hybrid.sp.to_numpy(copy=False)
    A = _hybrid.A
    B = _hybrid.B

    res = array.geopotential_on_hybrid_levels(t_arr, q_arr, zs_arr, sp_arr, A=A, B=B, alpha_top=alpha_top)

    return _hybrid.to_fieldlist(res, template=t[0], param_name="geopotential")


def height_on_hybrid_levels(
    t: FieldList,
    q: FieldList,
    zs: FieldList | Field,
    sp: FieldList | Field,
    A: ArrayLike | None = None,
    B: ArrayLike | None = None,
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
    from .hybrid import _HybridInput

    _hybrid = _HybridInput()
    _hybrid.add_sp(sp)
    _hybrid.add_zs(zs)
    _hybrid.add_t(t)
    _hybrid.add_q(q)
    _hybrid.generate_AB(A, B)
    _hybrid.check_levels()  # check that all input FieldLists have the same levels and return the levels

    t_arr = _hybrid.t.to_numpy(copy=False)
    q_arr = _hybrid.q.to_numpy(copy=False)
    zs_arr = _hybrid.zs.to_numpy(copy=False)
    sp_arr = _hybrid.sp.to_numpy(copy=False)
    A = _hybrid.A
    B = _hybrid.B

    res = array.height_on_hybrid_levels(
        t_arr, q_arr, zs_arr, sp_arr, A=A, B=B, alpha_top=alpha_top, h_type=h_type, h_reference=h_reference
    )

    return _hybrid.to_fieldlist(res, template=t[0], param_name="height")


def interpolate_hybrid_to_pressure_levels(
    data: FieldList,
    target_p: ArrayLike,
    sp: FieldList | Field,
    A: ArrayLike | None = None,
    B: ArrayLike | None = None,
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
        full-level. For a given model level only one field is allowed. The fields do not
        need to sorted in any particular order. When the resulting fields are created, their
        metadata is copied from the field with the lowest model level number.
    target_p: ArrayLike
        Target pressure levels (Pa).
    A: ArrayLike | None
        A-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
    B: ArrayLike | None
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
    from .hybrid import _HybridInput, to_fieldlist

    _hybrid = _HybridInput()
    _hybrid.add_sp(sp)
    _hybrid.add_profile(data, "data")
    _hybrid.generate_AB(A, B)
    _hybrid.check_levels()  # check that all input FieldLists have the same levels and return the levels

    data_arr = _hybrid.data.to_numpy(copy=False)
    sp_arr = _hybrid.sp.to_numpy(copy=False)
    A = _hybrid.A
    B = _hybrid.B

    res_arr = array.interpolate_hybrid_to_pressure_levels(
        data_arr,
        target_p,
        sp_arr,
        A,
        B,
        alpha_top,
        interpolation,
        aux_bottom_data,
        aux_bottom_p,
        aux_top_data,
        aux_top_p,
        vertical_dim=0,
    )

    return to_fieldlist(res_arr, template=data[0], levels=target_p, vertical={"level_type": "pressure"})


def interpolate_hybrid_to_height_levels(
    data: FieldList,
    target_h: ArrayLike,
    t: FieldList,
    q: FieldList,
    zs: FieldList | Field,
    sp: FieldList | Field,
    A: ArrayLike | None = None,
    B: ArrayLike | None = None,
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
    from .hybrid import _HybridInput, to_fieldlist

    _hybrid = _HybridInput()
    _hybrid.add_sp(sp)
    _hybrid.add_zs(zs)
    _hybrid.add_t(t)
    _hybrid.add_q(q)
    _hybrid.add_profile(data, "data")
    _hybrid.generate_AB(A, B)
    _hybrid.check_levels()  # check that all input FieldLists have the same levels and return the levels

    data_arr = _hybrid.data.to_numpy(copy=False)
    t_arr = _hybrid.t.to_numpy(copy=False)
    q_arr = _hybrid.q.to_numpy(copy=False)
    zs_arr = _hybrid.zs.to_numpy(copy=False)
    sp_arr = _hybrid.sp.to_numpy(copy=False)
    A = _hybrid.A
    B = _hybrid.B

    res_arr = array.interpolate_hybrid_to_height_levels(
        data_arr,
        target_h,
        t_arr,
        q_arr,
        zs_arr,
        sp_arr,
        A,
        B,
        alpha_top=alpha_top,
        interpolation=interpolation,
        h_type=h_type,
        h_reference=h_reference,
        aux_bottom_data=aux_bottom_data,
        aux_bottom_h=aux_bottom_h,
        aux_top_data=aux_top_data,
        aux_top_h=aux_top_h,
        vertical_dim=0,
    )

    return to_fieldlist(res_arr, template=data[0], levels=target_h, vertical={"level_type": "height"})


def interpolate_pressure_to_height_levels(
    data: FieldList,
    target_h: ArrayLike,
    z: FieldList,
    zs: FieldList | Field = None,
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
