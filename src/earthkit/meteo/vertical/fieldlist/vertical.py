# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import Literal

from earthkit.data import Field, FieldList  # type: ignore[import]
from numpy.typing import ArrayLike

from earthkit.meteo import constants
from earthkit.meteo.utils.decorators import fieldlist_ufunc
from earthkit.meteo.utils.fieldlist import get_hybrid_level_parameters
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
    the Earth (see :py:attr:`earthkit.meteo.constants.g`).
    """
    fieldlist_ufunc_kwargs = {"default_variable": "geopotential_height"}

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
    the Earth (see :py:attr:`earthkit.meteo.constants.g`).
    """
    fieldlist_ufunc_kwargs = {"default_variable": "geopotential"}

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
    (see :py:attr:`earthkit.meteo.constants.R_earth`).
    """
    fieldlist_ufunc_kwargs = {"default_variable": "geopotential_height"}

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
          (see :py:attr:`earthkit.meteo.constants.R_earth`)
        * :math:`g` is the gravitational acceleration on the surface of
          the Earth (see :py:attr:`earthkit.meteo.constants.g`)
    """
    fieldlist_ufunc_kwargs = {"default_variable": "geopotential"}

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
    (see :py:attr:`earthkit.meteo.constants.R_earth`).
    """
    fieldlist_ufunc_kwargs = {"default_variable": "geometric_height_above_sea"}

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
          (see :py:attr:`earthkit.meteo.constants.R_earth`)
        * :math:`g` is the gravitational acceleration on the surface of
          the Earth (see :py:attr:`earthkit.meteo.constants.g`)
    """
    fieldlist_ufunc_kwargs = {"default_variable": "geometric_height_above_sea"}

    return fieldlist_ufunc(
        array.geometric_height_from_geopotential,
        z,
        R_earth=R_earth,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
    )


def pressure_on_hybrid_levels(
    sp: FieldList | Field,
    A: ArrayLike | None = None,
    B: ArrayLike | None = None,
    levels: ArrayLike | None = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    output: Literal["full", "half", "delta", "alpha"] | list | tuple = "full",
) -> FieldList | tuple[FieldList, ...]:
    r"""Compute pressure and related parameters on hybrid (IFS model) levels.

    Parameters
    ----------
    sp: FieldList|Field
        Surface pressure (Pa). Can be a single Field or a FieldList. If a FieldList is
        provided, it must contain exactly one Field.
    A: ArrayLike | None, optional
        A-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number
        (from the top of the atmosphere toward the surface). When None (default), the A
        and B coefficients will be inferred from the metadata of the input field ``sp``.
    B: ArrayLike | None, optional
        B-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        (from the top of the atmosphere toward the surface).
        Must have the same size as ``A``. When None (default), the A and B coefficients
        will be inferred from the metadata of the input field ``sp``.
    levels: ArrayLike | None, optional
        Hybrid full-levels to return. Level numbering starts at 1 at the top
        of the atmosphere and increases towards the surface.  If None (default), all the levels are
        returned in the order defined by the A and B coefficients (i.e. ascending order
        with respect to the model level number). If only half-levels are requested in ``output``
        the ``levels`` are interpreted as half-level numbers (so 0 is a valid half-level
        number corresponding to the top of the atmosphere).
    alpha_top: {"ifs", "arpege"}, default="ifs"
        Option to initialise the alpha parameter on the top of the model
        atmosphere. See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.
    output : {"full", "half", "delta", "alpha"} | list | tuple, default="full"
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

    Notes
    -----
    The hybrid model levels divide the atmosphere into :math:`NLEV` layers. These layers are defined
    by the pressures at the interfaces between them for :math:`0 \leq k \leq NLEV`, which are
    the half-levels :math:`p_{k+1/2}` (indices increase from the top of the atmosphere towards
    the surface). The half-levels are defined by the ``A`` and ``B`` coefficients in such a way
    that at the top of the atmosphere the first half-level pressure :math:`p_{0+1/2}` is a constant,
    while at the surface :math:`p_{NLEV+1/2}` is the surface pressure.

    The full-level pressure :math:`p_{k}` associated with each model
    level is defined as the middle of the layer for :math:`1 \leq k \leq NLEV`.

    The level definitions can be written as:

    .. math::

        p_{k+1/2} = A_{k+1/2} + p_{s}  B_{k+1/2}  \quad k=0, 1, ..., NLEV

        p_{k} = \frac{1}{2}  (p_{k-1/2} + p_{k+1/2})  \quad k=1, 2, ..., NLEV

    where

        - :math:`p_{s}` is the surface pressure
        - :math:`p_{k+1/2}` is the pressure at the half-levelss
        - :math:`p_{k}` is the pressure at the full-levels
        - :math:`A_{k+1/2}` and :math:`B_{k+1/2}` are the A- and B-coefficients defining
          the model levels.

    For more details see [IFS-CY49R1-Dynamics]_ Chapter 2, Section 2.2.1.

    Examples
    --------
    - :ref:`/tutorials/vertical/hybrid_levels_fieldlist.ipynb`

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
        sp_values = field.to_numpy(copy=False)
        res_values = array.pressure_on_hybrid_levels(
            sp_values,
            A,
            B,
            levels=levels,
            alpha_top=alpha_top,
            output=output,
        )

        assert len(res_values) >= 2

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
        Temperature on hybrid full-levels (K). Fields must correspond to a distinct
        set of levels in arbitrary order. Not
        all the levels must be present, but a contiguous level range including the bottom-most
        level must be used. E.g. if the vertical coordinate system has 137 model levels using
        only a subset of levels between e.g. 137-96 is allowed.
    q: FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same
        number of fields and levels as ``t``, the level ordering can be different.
    alpha: FieldList
        Alpha parameter computed using
        :func:`pressure_on_hybrid_levels`. Must have the same number of
        fields and levels as ``t``, the level ordering can be different.
    delta: FieldList
        Delta parameter computed using
        :func:`pressure_on_hybrid_levels`. Must have the same number of
        fields and levels as ``t``, the level ordering can be different.

    Returns
    -------
    FieldList
        Geopotential thickness (m2/s2) between the surface and hybrids
        full-levels.

    See Also
    --------
    pressure_on_hybrid_levels
    earthkit.meteo.vertical.array.relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta

    Examples
    --------
    - :ref:`/tutorials/vertical/hybrid_levels_fieldlist.ipynb`

    """
    from .utils import HybridData

    source = HybridData()
    source.add_t(t)
    source.add_q(q)
    source.add_alpha(alpha)
    source.add_delta(delta)
    source.check_levels()  # check that all input FieldLists have the same levels and return the levels

    t_arr = source.t.to_numpy(copy=False)
    q_arr = source.q.to_numpy(copy=False)
    alpha_arr = source.alpha.to_numpy(copy=False)
    delta_arr = source.delta.to_numpy(copy=False)

    res_arr = array.relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta(
        t=t_arr,
        q=q_arr,
        alpha=alpha_arr,
        delta=delta_arr,
    )

    return source.to_fieldlist(res_arr, template=t[0], param_name="relative_geopotential_thickness")


def relative_geopotential_thickness_on_hybrid_levels(
    t: FieldList,
    q: FieldList,
    sp: FieldList | Field,
    A: ArrayLike | None = None,
    B: ArrayLike | None = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
) -> FieldList:
    r"""Compute the geopotential thickness between the surface and hybrid full-levels.

    Parameters
    ----------
    t: FieldList
        Temperature on hybrid full-levels (K). Fields must correspond to a
        distinct set of levels in arbitrary order. Not
        all the levels must be present, but a contiguous level range including the bottom-most
        level must be used. E.g. if the vertical coordinate system has 137 model levels using
        only a subset of levels between e.g. 137-96 is allowed.
    q: FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same
        number of fields and levels as ``t``, the level ordering can be different.
    sp: FieldList|Field
        Surface pressure (Pa). Can be a single Field or a FieldList. If a FieldList
        is provided, it must contain exactly one Field.
    A: ArrayLike|None, optional
        A-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number. If None,
        the A and B coefficients will be inferred from the metadata of the input fields
        ``sp``, ``t`` and ``q`` (tried in this order).
    B: ArrayLike|None, optional
        B-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        Must be defined when ``A`` is provided and have the same size as ``A``.
        If None, the A and B coefficients will be inferred from the metadata of the input fields
        ``sp``, ``t`` and ``q`` (tried in this order).
    alpha_top: {"ifs", "arpege"}, default="ifs"
        Option to initialise the alpha parameter on the top of the model
        atmosphere. See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.

    Returns
    -------
    FieldList
        Geopotential thickness (m2/s2) between the surface and hybrid
        full-levels. The fields in the output FieldList are sorted by
        their hybrid level number in ascending order (from the top of the atmosphere
        towards the surface).

    See Also
    --------
    pressure_on_hybrid_levels
    relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta
    earthkit.meteo.vertical.array.relative_geopotential_thickness_on_hybrid_levels

    Notes
    -----
    ``alpha`` and ``delta`` can be calculated using :func:`pressure_on_hybrid_levels`.

    The computations are described in [IFS-CY49R1-Dynamics]_ Chapter 2, Section 2.2.1.

    Examples
    --------
    - :ref:`/tutorials/vertical/hybrid_levels_fieldlist.ipynb`

    """
    from .utils import HybridData

    source = HybridData()
    source.add_t(t)
    source.add_q(q)
    source.add_sp(sp)
    source.generate_AB(A, B)
    source.check_levels()  # check that all input FieldLists have the same levels and return the levels

    t_arr = source.t.to_numpy(copy=False)
    q_arr = source.q.to_numpy(copy=False)
    sp_arr = source.sp.to_numpy(copy=False)
    A = source.A
    B = source.B

    res_arr = array.relative_geopotential_thickness_on_hybrid_levels(
        t=t_arr,
        q=q_arr,
        sp=sp_arr,
        A=A,
        B=B,
        alpha_top=alpha_top,
    )

    return source.to_fieldlist(res_arr, template=t[0], param_name="relative_geopotential_thickness")


def geopotential_on_hybrid_levels(
    t: FieldList,
    q: FieldList,
    zs: FieldList | Field,
    sp: FieldList | Field,
    A: ArrayLike | None = None,
    B: ArrayLike | None = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
) -> FieldList:
    r"""Compute geopotential on hybrid (IFS model) full-levels.

    Parameters
    ----------
    t: FieldList
        Temperature on hybrid full-levels (K). Fields must correspond to a
        distinct set of hybrid full-levels in arbitrary order. Not
        all the levels must be present, but a contiguous level range including the bottom-most
        level must be used. E.g. if the vertical coordinate system has 137 model levels using
        only a subset of levels between e.g. 137-96 is allowed.
    q: FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same
        number of fields and level levels as ``t``, the level ordering can be different.
    zs: FieldList|Field
        Surface geopotential (m2/s2). Can be a single Field or a FieldList. If a FieldList
        is provided, it must contain exactly one Field.
    sp: FieldList|Field
        Surface pressure (Pa). Can be a single Field or a FieldList. If a FieldList
        is provided, it must contain exactly one Field.
    A: ArrayLike|None, optional
        A-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        When None, the A and B coefficients will be inferred from the metadata of the input fields
        ``sp``, ``zs``, ``t`` and ``q`` (tried in this order).
    B: ArrayLike| None, optional
        B-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        Must be defined when ``A`` is provided and have the same size as ``A``.
        When None, the A and B coefficients will be inferred from the metadata of the input fields
        ``sp``, ``zs``, ``t`` and ``q`` (tried in this order).
    alpha_top: {"ifs", "arpege"}, default="ifs"
        Option to initialise the alpha parameter on the top of the model
        atmosphere. See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.

    Returns
    -------
    FieldList
        Geopotential (m2/s2) on hybrid full-levels. The fields in the output FieldList are sorted by
        their hybrid level number in ascending order (from the top of the atmosphere
        towards the surface).

    See Also
    --------
    pressure_on_hybrid_levels
    relative_geopotential_thickness_on_hybrid_levels
    earthkit.meteo.vertical.array.geopotential_on_hybrid_levels


    Notes
    -----
    The computations are described in [IFS-CY49R1-Dynamics]_ Chapter 2, Section 2.2.1.

    Examples
    --------
    - :ref:`/tutorials/vertical/hybrid_levels_fieldlist.ipynb`

    """
    from .utils import HybridData

    source = HybridData()
    source.add_sp(sp)
    source.add_zs(zs)
    source.add_t(t)
    source.add_q(q)
    source.generate_AB(A, B)
    source.check_levels()  # check that all input FieldLists have the same levels and return the levels

    t_arr = source.t.to_numpy(copy=False)
    q_arr = source.q.to_numpy(copy=False)
    zs_arr = source.zs.to_numpy(copy=False)
    sp_arr = source.sp.to_numpy(copy=False)
    A = source.A
    B = source.B

    res = array.geopotential_on_hybrid_levels(t_arr, q_arr, zs_arr, sp_arr, A=A, B=B, alpha_top=alpha_top)

    return source.to_fieldlist(res, template=t[0], param_name="geopotential")


def height_on_hybrid_levels(
    t: FieldList,
    q: FieldList,
    zs: FieldList | Field,
    sp: FieldList | Field,
    A: ArrayLike | None = None,
    B: ArrayLike | None = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    h_type: Literal["geometric", "geopotential"] = "geometric",
    h_reference: Literal["ground", "sea"] = "ground",
) -> FieldList:
    r"""Compute the height on hybrid (IFS model) full-levels.

    Parameters
    ----------
    t: FieldList
        Temperature on hybrid full-levels (K). Fields must correspond to a
        distinct set of hybrid full-levels in arbitrary order. Not
        all the levels must be present, but a contiguous level range including the bottom-most
        level must be used. E.g. if the vertical coordinate system has 137 model levels using
        only a subset of levels between e.g. 137-96 is allowed.
    q: FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same
        number of fields and levels as ``t``, the level ordering can be different.
    zs: FieldList|Field
        Surface geopotential (m2/s2). Can be a single Field or a FieldList. If a FieldList
        is provided, it must contain exactly one Field.
        Not used when ``h_type`` is ``"geopotential"`` and ``h_reference`` is ``"ground"``.
    sp: FieldList|Field
        Surface pressure (Pa). Can be a single Field or a FieldList. If a FieldList
        is provided, it must contain exactly one Field.
    A: ArrayLike|None, optional
        A-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        When None, the A and B coefficients will be inferred from the metadata of the input fields
        ``sp``, ``zs``, ``t`` and ``q`` (tried in this order).
    B: ArrayLike|None, optional
        B-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        Must be defined when ``A`` is provided and have the same size as ``A``. When None, the
        A and B coefficients will be inferred from the metadata of the input fields
        ``sp``, ``zs``, ``t`` and ``q`` (tried in this order).
    alpha_top: {"ifs", "arpege"}, default="ifs"
        Option to initialise the alpha parameter on the top of the model
        atmosphere. See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.
    h_type: {"geometric", "geopotential"}, default="geometric"
        Type of height to compute. Default is ``"geometric"``. Possible values:

        - ``"geometric"``: geometric height (m)
        - ``"geopotential"``: geopotential height (m)

    h_reference: {"ground", "sea"}, default="ground"
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

    Notes
    -----
    The height is calculated from the geopotential on hybrid levels, which is computed
    from the ``t``, ``q``, ``zs`` and the hybrid
    level definition (``A``, ``B``  or ``sp``). The
    computations are described in [IFS-CY49R1-Dynamics]_ Chapter 2, Section 2.2.1.

    Examples
    --------
    - :ref:`/tutorials/vertical/hybrid_levels_fieldlist.ipynb`

    """
    from .utils import HybridData

    source = HybridData()
    source.add_sp(sp)
    source.add_zs(zs)
    source.add_t(t)
    source.add_q(q)
    source.generate_AB(A, B)
    source.check_levels()  # check that all input FieldLists have the same levels and return the levels

    t_arr = source.t.to_numpy(copy=False)
    q_arr = source.q.to_numpy(copy=False)
    zs_arr = source.zs.to_numpy(copy=False)
    sp_arr = source.sp.to_numpy(copy=False)
    A = source.A
    B = source.B

    res = array.height_on_hybrid_levels(
        t_arr, q_arr, zs_arr, sp_arr, A=A, B=B, alpha_top=alpha_top, h_type=h_type, h_reference=h_reference
    )

    if h_type == "geopotential" and h_reference == "ground":
        param_name = "geopotential_height_above_ground"
    elif h_type == "geopotential" and h_reference == "sea":
        param_name = "geopotential_height_above_sea"
    elif h_type == "geometric" and h_reference == "ground":
        param_name = "geometric_height_above_ground"
    elif h_type == "geometric" and h_reference == "sea":
        param_name = "geometric_height_above_sea"

    return source.to_fieldlist(res, template=t[0], param_name=param_name)


def interpolate_hybrid_to_pressure_levels(
    data: FieldList,
    target_p: ArrayLike | FieldList | Field,
    sp: FieldList | Field,
    A: ArrayLike | None = None,
    B: ArrayLike | None = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    interpolation: Literal["linear", "log", "nearest"] = "linear",
    aux_bottom_data: float | FieldList | Field | None = None,
    aux_bottom_p: float | FieldList | Field | None = None,
    aux_top_data: float | FieldList | Field | None = None,
    aux_top_p: float | FieldList | Field | None = None,
) -> FieldList:
    r"""Interpolate data from hybrid full-levels to pressure levels.

    Parameters
    ----------
    data: FieldList
        Data on hybrid full-levels to be interpolated. Fields must correspond to a
        distinct set of hybrid full-levels in arbitrary order. Not
        all the levels must be present, but a contiguous level range including the bottom-most
        level must be used. E.g. if the vertical coordinate system has 137 model levels using
        only a subset of levels between e.g. 137-96 is allowed.
        Must have at least two fields.
    target_p: ArrayLike | FieldList | Field
        Target pressures(s) (Pa) to which ``data`` will be interpolated.
        When provided as an ArrayLike, it must be a 1D array of pressures
        each defining a constant target (a single number is also allowed in
        this case). When it is a
        FieldList or Field the field values themselves will provide the target pressures to the
        corresponding grid points in ``data`` (and not the level metadata).
    sp: FieldList | Field
        Surface pressure (Pa). Can be a single Field or a FieldList. If a FieldList
        is provided, it must contain exactly one Field.
    A: ArrayLike | None, optional
        A-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        When None, the A and B coefficients will be inferred from the metadata of the input fields
        ``sp`` and ``data`` (tried in this order).
    B: ArrayLike | None, optional
        B-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        Must be defined when ``A`` is provided and have the same size as ``A``. When None, the
        A and B coefficients will be inferred from the metadata of the input fields
        ``sp`` and ``data`` (tried in this order).
    alpha_top: {"ifs", "arpege"}, default="ifs"
        Option to initialise the alpha parameter on the top of the model
        atmosphere. See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.
    interpolation: {"linear", "log", "nearest"}, default="linear"
        Interpolation mode. Possible values:

        - ``"linear"``: linear interpolation in pressure
        - ``"log"``: linear interpolation in log-pressure
        - ``"nearest"``: nearest level interpolation

    aux_bottom_data: float | FieldList | Field | None, optional
        Auxiliary data for interpolation to targets below the bottom hybrid full-level
        and above the level specified by ``aux_bottom_p``. Can be a number, a single Field
        or a FieldList containing exactly one Field.  Must be provided together with ``aux_bottom_p``.
    aux_bottom_p: float | FieldList | Field | None, optional
        Pressure(s) (Pa) of ``aux_bottom_data``. Can be a number, a single Field, or a FieldList
        containing exactly one Field.
    aux_top_data: float | FieldList | Field | None, optional
        Auxiliary data for interpolation to targets above the top hybrid full-level
        and below the level specified by ``aux_top_p``. Can be a number, a single Field or a
        FieldList containing exactly one Field.  Must be provided together with ``aux_top_p``.
    aux_top_p: float | FieldList | Field | None, optional
        Pressure(s) (Pa) of ``aux_top_data``. Can be a number, a single Field, or a FieldList
        containing exactly one Field.

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

    Examples
    --------
    - :ref:`/tutorials/vertical/interpolate_hybrid_to_pl_fieldlist.ipynb`

    """
    from .utils import HybridData, SingleVariable, TargetVariable, to_resulting_fieldlist

    if aux_bottom_data is not None or aux_bottom_p is not None:
        if aux_bottom_data is None or aux_bottom_p is None:
            raise ValueError("Both aux_bottom_data and aux_bottom_p must be provided together.")

    if aux_top_data is not None or aux_top_p is not None:
        if aux_top_data is None or aux_top_p is None:
            raise ValueError("Both aux_top_data and aux_top_p must be provided together.")

    # prepare input data
    source = HybridData()
    source.add_sp(sp)
    source.add_profile(key="data", fl=data)
    source.generate_AB(A, B)
    source.check_levels()  # check that all input FieldLists have the same levels and return the levels

    # prepare target data
    target = TargetVariable.build(
        key="target_coord",
        fl=target_p,
    )

    # get arrays
    data_arr = source.data.to_numpy(copy=False)
    sp_arr = source.sp.to_numpy(copy=False)
    A = source.A
    B = source.B

    target_p_arr = target.fl.to_numpy(copy=False)

    aux = {
        "aux_bottom_data": aux_bottom_data,
        "aux_bottom_p": aux_bottom_p,
        "aux_top_data": aux_top_data,
        "aux_top_p": aux_top_p,
    }
    for key in aux:
        if aux[key] is not None:
            v = SingleVariable.build(key=key, fl=aux[key], fl_template=source.sp).fl.to_numpy(copy=False)
            aux[key] = v

    res_arr = array.interpolate_hybrid_to_pressure_levels(
        data_arr,
        target_p_arr,
        sp_arr,
        A=A,
        B=B,
        alpha_top=alpha_top,
        interpolation=interpolation,
        **aux,
        vertical_dim=0,
    )

    levels = target.first_field_values()
    return to_resulting_fieldlist(res_arr, template=data[0], levels=levels, vertical={"level_type": "pressure"})


def interpolate_hybrid_to_height_levels(
    data: FieldList,
    target_h: ArrayLike | FieldList | Field,
    t: FieldList,
    q: FieldList,
    zs: FieldList | Field | None,
    sp: FieldList | Field,
    A: ArrayLike | None = None,
    B: ArrayLike | None = None,
    alpha_top: Literal["ifs", "arpege"] = "ifs",
    h_type: Literal["geometric", "geopotential"] = "geometric",
    h_reference: Literal["ground", "sea"] = "ground",
    interpolation: Literal["linear", "log", "nearest"] = "linear",
    aux_bottom_data: float | FieldList | Field | None = None,
    aux_bottom_h: float | FieldList | Field | None = None,
    aux_top_data: float | FieldList | Field | None = None,
    aux_top_h: float | FieldList | Field | None = None,
) -> FieldList:
    r"""Interpolate data from hybrid full-levels to height levels.

    Parameters
    ----------
    data: FieldList
        Data to be interpolated. Fields must correspond to a
        distinct set of hybrid full-levels in arbitrary order.
        Not all the levels must be present, but a contiguous level range including the bottom-most
        level must be used. E.g. if the vertical coordinate system has 137 model levels using
        only a subset of levels between e.g. 137-96 is allowed.
        Must have at least two fields.
    target_h: ArrayLike | FieldList | Field
        Target height(s) (m) to which ``data`` will be interpolated.
        When provided as an ArrayLike, it must be a 1D array of heights
        each defining a constant target (a single number is also allowed in
        this case). When it is a
        FieldList or Field the field values themselves will provide the target heights to the
        corresponding grid points in ``data`` (and not the level metadata). The type
        and reference of the height are defined by ``h_type`` and ``h_reference``.
    t: FieldList
        Temperature on hybrid full-levels (K). Must have the same number of
        fields and levels as ``data``, the level ordering can be different.
    q: FieldList
        Specific humidity on hybrid full-levels (kg/kg). Must have the same
        number of fields and levels as ``data``, the level ordering can be different.
    zs: FieldList|Field|None
        Surface geopotential (m2/s2). Can be a single Field or a FieldList. If a FieldList
        is provided, it must contain exactly one Field. Not used  when ``h_type`` is
        "geopotential" and ``h_reference`` is "ground".
    sp: FieldList|Field
        Surface pressure (Pa). Can be a single Field or a FieldList. If a FieldList
        is provided, it must contain exactly one Field.
    A: ArrayLike | None, optional
        A-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number. When None,
        the A and B coefficients will be inferred from the metadata of the input fields
        ``sp``, ``zs``, ``t`` and ``q`` (tried in this order).
    B: ArrayLike | None, optional
        B-coefficients defining the hybrid levels. Must contain all the
        half-levels in ascending order with respect to the model level number.
        Must be defined when ``A`` is provided and have the same size as ``A``. When None, the
        A and B coefficients will be inferred from the metadata of the input fields
        ``sp``, ``zs``, ``t`` and ``q`` (tried in this order).
    alpha_top: {"ifs", "arpege"}, default="ifs"
        Option to initialise the alpha parameter on the top of the model
        atmosphere. See :func:`earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
        for details.
    h_type: {"geometric", "geopotential"}, default="geometric"
        Type of height to compute. Default is ``"geometric"``. Possible values:

        - ``"geometric"``: geometric height (m)
        - ``"geopotential"``: geopotential height (m)

    h_reference: {"ground", "sea"}, default="ground"
        Reference level for the height calculation. Default is ``"ground"``.
        Possible values:

        - ``"ground"``: height with respect to the ground/surface level
        - ``"sea"``: height with respect to the sea level

    interpolation: {"linear", "log", "nearest"}, default="linear"
        Interpolation mode. Default is ``"linear"``. Possible values:

        - ``"linear"``: linear interpolation in height
        - ``"log"``: linear interpolation in log-height
        - ``"nearest"``: nearest level interpolation

    aux_bottom_data: float|FieldList|Field|None, optional
        Auxiliary data for interpolation to heights between the bottom hybrid full-level
        and ``aux_bottom_h``. Can be a number, a single Field or a FieldList containing exactly one Field.
        Must be provided together with ``aux_bottom_h``.
    aux_bottom_h: float|FieldList|Field|None, optional
        Heights (m) of ``aux_bottom_data``. Can be a number, a single Field or a FieldList
        containing exactly one Field.
    aux_top_data: float|FieldList|Field|None, optional
        Auxiliary data for interpolation to heights above the top hybrid full-level
        and below ``aux_top_h``. Can be a number, a single Field or a FieldList containing
        exactly one Field. Must be provided together with ``aux_top_h``.
    aux_top_h: float|FieldList|Field|None, optional
        Heights (m) of ``aux_top_data``. Can be a number, a single Field or a FieldList
        containing exactly one Field.

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

    Examples
    --------
    - :ref:`/tutorials/vertical/interpolate_hybrid_to_hl_fieldlist.ipynb`

    """
    from .utils import HybridData, SingleVariable, TargetVariable, to_resulting_fieldlist

    if aux_bottom_data is not None or aux_bottom_h is not None:
        if aux_bottom_data is None or aux_bottom_h is None:
            raise ValueError("Both aux_bottom_data and aux_bottom_h must be provided together.")

    if aux_top_data is not None or aux_top_h is not None:
        if aux_top_data is None or aux_top_h is None:
            raise ValueError("Both aux_top_data and aux_top_h must be provided together.")

    # prepare input data
    source = HybridData()
    source.add_sp(sp)
    source.add_zs(zs)
    source.add_t(t)
    source.add_q(q)
    source.add_profile(key="data", fl=data)
    source.generate_AB(A, B)
    source.check_levels()  # check that all input FieldLists have the same levels and return the levels

    # prepare target
    target = TargetVariable.build(
        key="target_coord",
        fl=target_h,
    )

    # get arrays
    data_arr = source.data.to_numpy(copy=False)
    t_arr = source.t.to_numpy(copy=False)
    q_arr = source.q.to_numpy(copy=False)
    zs_arr = source.zs.to_numpy(copy=False)
    sp_arr = source.sp.to_numpy(copy=False)
    A = source.A
    B = source.B

    target_h_arr = target.fl.to_numpy(copy=False)

    aux = {
        "aux_bottom_data": aux_bottom_data,
        "aux_bottom_h": aux_bottom_h,
        "aux_top_data": aux_top_data,
        "aux_top_h": aux_top_h,
    }
    for key in aux:
        if aux[key] is not None:
            v = SingleVariable.build(key=key, fl=aux[key], fl_template=source.sp).fl.to_numpy(copy=False)
            aux[key] = v

    res_arr = array.interpolate_hybrid_to_height_levels(
        data_arr,
        target_h_arr,
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
        **aux,
        vertical_dim=0,
    )

    if h_reference == "ground":
        vertical = {"level_type": "height_above_ground_level"}
    else:
        vertical = {"level_type": "height_above_mean_sea_level"}

    levels = target.first_field_values()
    return to_resulting_fieldlist(res_arr, template=data[0], levels=levels, vertical=vertical)


def interpolate_pressure_to_height_levels(
    data: FieldList,
    target_h: ArrayLike | FieldList | Field,
    z: FieldList,
    zs: FieldList | Field | None = None,
    h_type: Literal["geometric", "geopotential"] = "geometric",
    h_reference: Literal["ground", "sea"] = "ground",
    interpolation: Literal["linear", "log", "nearest"] = "linear",
    aux_bottom_data: float | FieldList | Field | None = None,
    aux_bottom_h: float | FieldList | Field | None = None,
    aux_top_data: float | FieldList | Field | None = None,
    aux_top_h: float | FieldList | Field | None = None,
) -> FieldList:
    r"""Interpolate data from pressure levels to height levels.

    Parameters
    ----------
    data: FieldList
        Data on pressure levels to be interpolated. Fields must correspond to a
        distinct set of pressure levels in arbitrary order. Must have at least two fields.
    target_h: ArrayLike | FieldList | Field
        Target height levels (m). The type and reference of the height are
        defined by ``h_type`` and ``h_reference``.
    z: FieldList
        Geopotential (m2/s2) on the same pressure levels as ``data``. The number of fields
        and levels must be the same as in ``data``, but the level ordering can be different.
    zs: FieldList|Field|None, optional
        Surface geopotential (m2/s2). Can be a single Field or a FieldList. If a FieldList
        is provided, it must contain exactly one Field.
        Only used when ``h_reference`` is "ground".
    h_type: {"geometric", "geopotential"}, default="geometric"
        Type of height to compute. Default is ``"geometric"``. Possible values:

        - ``"geometric"``: geometric height (m)
        - ``"geopotential"``: geopotential height (m)

    h_reference: {"ground", "sea"}, default="ground"
        Reference level for the height calculation. Default is ``"ground"``.
        Possible values:

        - ``"ground"``: height with respect to the ground/surface level
        - ``"sea"``: height with respect to the sea level

    interpolation: {"linear", "log", "nearest"}, default="linear"
        Interpolation mode. Default is ``"linear"``. Possible values:

        - ``"linear"``: linear interpolation in height
        - ``"log"``: linear interpolation in log-height
        - ``"nearest"``: nearest level interpolation

    aux_bottom_data: float | FieldList | Field | None
        Auxiliary data for interpolation below the bottom pressure level.
    aux_bottom_h: float | FieldList | Field | None, optional
        Heights (m) of ``aux_bottom_data``.
    aux_top_data: float | FieldList | Field | None, optional
        Auxiliary data for interpolation above the top pressure level.
    aux_top_h: float | FieldList | Field | None, optional
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

    Examples
    --------
    - :ref:`/tutorials/vertical/interpolate_pl_to_hl_fieldlist.ipynb`

    """
    from .utils import MonotonicData, SingleVariable, TargetVariable, to_resulting_fieldlist

    if aux_bottom_data is not None or aux_bottom_h is not None:
        if aux_bottom_data is None or aux_bottom_h is None:
            raise ValueError("Both aux_bottom_data and aux_bottom_h must be provided together.")

    if aux_top_data is not None or aux_top_h is not None:
        if aux_top_data is None or aux_top_h is None:
            raise ValueError("Both aux_top_data and aux_top_h must be provided together.")

    # prepare input data
    source = MonotonicData(level_type="pressure", sort="ascending")
    if zs is not None:
        source.add_single(key="zs", name="Surface geopotential", fl=zs)
    source.add_profile(key="data", name="Data", fl=data)
    source.add_profile(key="z", name="Geopotential", fl=z)
    source.check_levels()  # check that all input FieldLists have the same levels and return the levels

    # prepare target
    target = TargetVariable.build(
        key="target_coord",
        fl=target_h,
    )

    # get arrays
    data_arr = source.data.to_numpy(copy=False)
    z_arr = source.z.to_numpy(copy=False)
    zs_arr = source.zs.to_numpy(copy=False) if zs is not None else None

    target_h_arr = target.fl.to_numpy(copy=False)

    aux = {
        "aux_bottom_data": aux_bottom_data,
        "aux_bottom_h": aux_bottom_h,
        "aux_top_data": aux_top_data,
        "aux_top_h": aux_top_h,
    }
    aux_template = source.data[0]
    for key in aux:
        if aux[key] is not None:
            v = SingleVariable.build(key=key, fl=aux[key], fl_template=aux_template).fl.to_numpy(copy=False)
            aux[key] = v

    res_arr = array.interpolate_pressure_to_height_levels(
        data_arr,
        target_h_arr,
        z_arr,
        zs=zs_arr,
        h_type=h_type,
        h_reference=h_reference,
        interpolation=interpolation,
        **aux,
        vertical_dim=0,
    )

    if h_reference == "ground":
        vertical = {"level_type": "height_above_ground_level"}
    else:
        vertical = {"level_type": "height_above_mean_sea_level"}

    levels = target.first_field_values()
    return to_resulting_fieldlist(res_arr, template=data[0], levels=levels, vertical=vertical)


def interpolate_monotonic(
    data: FieldList,
    coords: ArrayLike | FieldList | None = None,
    target_coords: ArrayLike | FieldList | Field | None = None,
    coord_type: str | None = None,
    interpolation: Literal["linear", "log", "nearest"] = "linear",
    aux_min_level_data: float | FieldList | Field | None = None,
    aux_min_level_coord: float | FieldList | Field | None = None,
    aux_max_level_data: float | FieldList | Field | None = None,
    aux_max_level_coord: float | FieldList | Field | None = None,
) -> FieldList:
    r"""Interpolate data between the same type of monotonic coordinate levels.

    Parameters
    ----------
    data: FieldList
        Data to be interpolated. Must have at least two fields.
    coords: ArrayLike | FieldList | None
        Vertical coordinates related to ``data``. A valid value must be
        provided. When it is a FieldList, it must have the same number of
        fields and levels as ``data``, but the level ordering can be different.
        The field values in ``coords`` define the vertical coordinate values for each
        corresponding field in ``data``. The level metadata
        is only used to pair up the fields in ``coords`` and ``data``.
        The vertical coordinate
        values defined in this way must be monotonic along the vertical
        axis when sorted by the level (either ascending or descending).
        When provided as an ArrayLike, it must be a 1D array with each value
        corresponding to the field at the same position in ``data``.
    target_coords: ArrayLike | FieldList | Field | None
        Target coordinate levels to which ``data`` will be interpolated. When it is a
        FieldList or Field each field value provides the coordinate values the ``data``
        will be interpolated to. When provided as an ArrayLike, it must be a 1D array of
        coordinate values each defining a constant target
        level. The values must be of the same type of coordinate as that of ``coords``.
    coord_type: str | None
        Type of the coordinate levels in ``coord`` and ``target_coords``.
        The possible values are level types supported in a Field in earthkit.data.
        See: :py:class:`~earthkit.data.field.component.level_type.LevelType` for details.
        A valid value must be provided.
    interpolation: {"linear", "log", "nearest"}, default="linear"
        Interpolation mode. Default is ``"linear"``. Possible values:

        - ``"linear"``: linear interpolation between the two nearest levels
        - ``"log"``: linear interpolation in logarithm of coordinate
        - ``"nearest"``: nearest level interpolation

    aux_min_level_data: float | FieldList | Field | None, optional
        Auxiliary data for interpolation to target levels below the minimum level
        of ``coord`` and above `aux_min_level_coord`. Can be a number, a single Field or
        a FieldList containing exactly one Field. Must be provided together
        with ``aux_min_level_coord``.
    aux_min_level_coord: ArrayLike | float | FieldList | Field | None, optional
        Coordinates of ``aux_min_level_data``. Can be a number, a single Field or
        a FieldList containing exactly one Field. Must be provided together
        with ``aux_min_level_data``.
    aux_max_level_data: float | FieldList | Field | None, optional
        Auxiliary data for interpolation to target levels above the maximum level
        of ``coord`` and below `aux_max_level_coord`. Can be a number, a single Field or
        a FieldList containing exactly one Field. Must be provided together
        with ``aux_max_level_coord``.
    aux_max_level_coord: ArrayLike | float | FieldList | Field | None, optional
        Coordinates of ``aux_max_level_data``. Can be a number, a single Field or
        a FieldList containing exactly one Field. Must be provided together
        with ``aux_max_level_data``.

    Returns
    -------
    FieldList
        Data interpolated to the target levels. When interpolation is not
        possible for a given target level, the corresponding output values
        are set to NaN. The metadata of the output fields are copied from the
        first field in ``data`` and updated with the target level type and value.

    See Also
    --------
    earthkit.meteo.vertical.array.interpolate_monotonic


    Examples
    --------
    - :ref:`/tutorials/vertical/interpolate_hybrid_to_hl_fieldlist.ipynb`
    - :ref:`/tutorials/vertical/interpolate_pl_to_pl_fieldlist.ipynb`

    """
    from .utils import MonotonicData, SingleVariable, TargetVariable, to_resulting_fieldlist

    if aux_min_level_data is not None or aux_min_level_coord is not None:
        if aux_min_level_data is None or aux_min_level_coord is None:
            raise ValueError("Both aux_min_level_data and aux_min_level_coord must be provided together.")

    if aux_max_level_data is not None or aux_max_level_coord is not None:
        if aux_max_level_data is None or aux_max_level_coord is None:
            raise ValueError("Both aux_max_level_data and aux_max_level_coord must be provided together.")

    if coord_type is None:
        raise ValueError("No coordinate type specified. Please specify coord_type explicitly.")

    # prepare input data
    source = MonotonicData(level_type=None, sort="descending")
    source.add_profile(key="data", fl=data)
    source.add_profile(key="coord", fl=coords, fl_template=source.data, coord=True)

    # prepare target
    target = TargetVariable.build(
        key="target_coord",
        fl=target_coords,
    )

    # get arrays
    data_arr = source.data.to_numpy(copy=False)
    coord_arr = source.coord.to_numpy(copy=False)
    target_coords_arr = target.fl.to_numpy(copy=False)

    aux = {
        "aux_min_level_data": aux_min_level_data,
        "aux_min_level_coord": aux_min_level_coord,
        "aux_max_level_data": aux_max_level_data,
        "aux_max_level_coord": aux_max_level_coord,
    }

    for key in aux:
        if aux[key] is not None:
            v = SingleVariable.build(key=key, fl=aux[key]).fl.to_numpy(copy=False)
            aux[key] = v

    res_arr = array.interpolate_monotonic(
        data_arr,
        coord_arr,
        target_coords_arr,
        interpolation=interpolation,
        **aux,
        vertical_dim=0,
    )

    levels = target.first_field_values()
    return to_resulting_fieldlist(res_arr, template=data[0], levels=levels, vertical={"level_type": coord_type})
