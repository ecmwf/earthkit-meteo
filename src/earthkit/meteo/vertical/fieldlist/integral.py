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

import numpy as np
from earthkit.data import Field, FieldList  # type: ignore[import]
from numpy.typing import NDArray

from earthkit.meteo.utils.fieldlist import pressure_from_metadata

from .. import array

# Metadata keys defining a single column: all the fields of a data column must share these
# values. The profiles holding the vertical coordinates (pressure, height, density) define
# the same column, but their parameter is not relevant.
_COLUMN_KEYS = ["parameter.variable", "time.base_datetime", "time.step", "ensemble.member"]
_COORDINATE_COLUMN_KEYS = _COLUMN_KEYS[1:]

# Vertical level types the geometric height can be taken from.
_HEIGHT_LEVEL_TYPES = ("height_above_ground_level", "height_above_mean_sea_level")

# Units that are not known. earthkit-data marks the units it cannot resolve with the
# "unknown" string (e.g. a GRIB parameter without units), and it parses the missing ones,
# None and "", as dimensionless. The units of a vertical integral can therefore never be
# left unset: when they cannot be computed from the ones of the input they are set to
# "unknown" to avoid keeping the units of the input, which are wrong.
_UNKNOWN_UNITS = "unknown"
_NO_UNITS = {"", _UNKNOWN_UNITS}


def _as_column(fl, name, keys=_COLUMN_KEYS):
    """Check that a FieldList holds a single vertical column and return it sorted by level.

    A column contains one field per vertical level: the fields share the same vertical
    level type, their levels are unique and the metadata keys listed in ``keys`` have the
    same value for every field.

    Parameters
    ----------
    fl: FieldList | Field
        Input to check. A single Field is accepted and wrapped into a FieldList.
    name: str
        Name of the argument, used in the error messages.
    keys: list of str
        Metadata keys identifying the column.

    Returns
    -------
    FieldList
        ``fl`` sorted by the vertical level in ascending order.

    Raises
    ------
    ValueError
        If ``fl`` is not a Field or a FieldList, if it is empty or if it does not hold a
        single column with uniquely levelled fields.
    """
    if isinstance(fl, Field):
        fl = FieldList.from_fields([fl])

    if not isinstance(fl, FieldList):
        raise ValueError(f'"{name}" must be a FieldList. Found {type(fl)}.')
    if len(fl) == 0:
        raise ValueError(f'"{name}" must contain at least one field.')

    u = fl.unique(["vertical.level", "vertical.level_type"] + keys)

    if len(u["vertical.level_type"]) != 1:
        raise ValueError(
            f'"{name}" must contain fields with a single vertical level type. '
            f"Found {len(u['vertical.level_type'])}: {u['vertical.level_type']}."
        )
    if len(u["vertical.level"]) != len(fl):
        raise ValueError(
            f'"{name}" must contain exactly one field for each vertical level. Found {len(fl)} fields '
            f"with {len(u['vertical.level'])} distinct levels."
        )

    for key in keys:
        values = u.get(key, ())
        if len(values) > 1:
            raise ValueError(
                f'"{name}" must contain a single vertical column. Found {len(values)} different '
                f'values for "{key}": {values}.'
            )

    return fl.order_by({"vertical.level": "ascending"})


def _levels(fl: FieldList) -> NDArray[Any]:
    """Return the vertical level values of a FieldList as a 1D float array."""
    return np.asarray(fl.get("vertical.level"), dtype=float)


def _profile_fieldlist_values(fl: FieldList, name: str, data: FieldList, interfaces: bool = True) -> NDArray[Any]:
    r"""Return the values of a profile FieldList matching ``data``, ordered like ``data``.

    The FieldList must hold the same number of fields as ``data``, defined on the same
    vertical levels, or (when ``interfaces`` is True) one more field, defined on the layer
    interfaces enclosing the data levels. In the latter case the fields are matched to the
    data by their order, e.g. the hybrid half-levels 0, ..., N enclose the hybrid
    full-levels 1, ..., N.

    Parameters
    ----------
    fl: FieldList | Field
        The profile to extract the values from.
    name: str
        Name of the argument, used in the error messages.
    data: FieldList
        The data column, already checked and sorted by :func:`_as_column`.
    interfaces: bool
        Whether ``fl`` may define the values on the layer interfaces.

    Returns
    -------
    ndarray
        The values of ``fl`` with the vertical axis first.

    Raises
    ------
    ValueError
        If ``fl`` does not hold a single column or if its number of fields or its levels
        do not match ``data``.
    """
    fl = _as_column(fl, name, keys=_COORDINATE_COLUMN_KEYS)
    nlev = len(data)
    n = len(fl)

    if interfaces and n == nlev + 1:
        # The profile defines the layer interfaces: they are matched to the data levels by
        # their order, their level values are not compared to the ones of the data.
        pass
    elif n == nlev:
        lev = _levels(fl)
        ref = _levels(data)
        if not np.allclose(lev, ref, rtol=1e-9, atol=1e-9):
            raise ValueError(
                f'Levels of "{name}" and "data" do not match. "{name}" is on {tuple(lev)} '
                f"while the levels of the input data are on {tuple(ref)}."
            )
    else:
        options = f"{nlev} (on the data levels) or {nlev + 1} (on the layer interfaces)" if interfaces else f"{nlev}"
        raise ValueError(
            f'"{name}" must contain {options} fields, to match the {nlev} level(s) of "data". Found {n} field(s).'
        )

    return fl.to_numpy(copy=False)


def _integral_fieldlist(values: NDArray[Any], template: Field, units: str) -> FieldList:
    """Wrap the result of a vertical integral into a FieldList containing a single field.

    The metadata is copied from ``template``, the vertical level type is set to
    "entire_atmosphere" (level 0) and the parameter units are extended with ``units``. The
    units of the result are set to :data:`_UNKNOWN_UNITS` and never left as they were in the
    input when the units of the input are not known or when earthkit-data does not accept
    the resulting unit string.
    """
    metadata = {
        "values": values,
        # The vertical component is replaced as a whole: setting only the level type would
        # keep the parametric coefficients of the input levels, which earthkit-data rejects.
        "vertical": {"level_type": "entire_atmosphere", "level": 0},
    }

    prefix = template.get("parameter.units")
    # the units of the input are not known
    if prefix is None or str(prefix).strip().lower() in _NO_UNITS:
        metadata["parameter.units"] = _UNKNOWN_UNITS
    else:
        metadata["parameter.units"] = f"{prefix} {units}"

    try:
        field = template.set(metadata)
    except ValueError:
        # earthkit-data could not parse the compound unit string: the units of the result
        # are not known. They cannot be left out: that would keep the ones of the input.
        metadata["parameter.units"] = _UNKNOWN_UNITS
        field = template.set(metadata)

    return FieldList.from_fields([field])


def vertical_integral_pressure(data: FieldList | Field, p: FieldList | Field | None = None) -> FieldList:
    r"""Integrate data over pressure.

    *New in version 1.4.0*

    Computes the integral of ``data`` over the vertical axis:

    .. math::

        I = \frac{1}{g} \int_{p_{top}}^{p_{bot}} f \, dp

    where :math:`f` are the data values, :math:`p` is the pressure (Pa) and :math:`g` is
    the gravitational acceleration on the surface of the Earth (see
    :py:attr:`earthkit.meteo.constants.g`). All the quantities are in SI units.

    Parameters
    ----------
    data: FieldList | Field
        Data to be integrated. It must hold a single column: one field per vertical level,
        with the same parameter, base date and time, step and ensemble member, unique
        levels and a single vertical level type. The fields are sorted by the level before
        the integration.
    p: FieldList | Field | None
        Pressure (Pa) defining the vertical coordinate. Either a FieldList with exactly one
        field for each level of ``data``, on the same levels, or a FieldList with one more
        field than ``data``, holding the pressure of the layer interfaces enclosing the
        data levels (e.g.
        :func:`earthkit.meteo.vertical.fieldlist.pressure_on_hybrid_levels` with
        ``output="half"``). The fields are sorted by their own level. When None (default)
        the pressure is taken from the level metadata of ``data``, which must be on
        pressure levels.

    Returns
    -------
    FieldList
        A FieldList with a single field: the vertically integrated data. Its metadata is
        copied from the first field of ``data``, the vertical level type is set to
        "entire_atmosphere" (level 0) and the units are the units of ``data`` extended
        with "kg m-2". They are set to "unknown" when the units of ``data`` are not known
        or when their combination is not accepted: the units of ``data`` are never kept.

    Raises
    ------
    ValueError
        If ``data`` or ``p`` does not hold a single column, if ``data`` is not on pressure
        levels and ``p`` is None or if the number of fields or the levels of ``p`` do not
        match ``data``.

    See Also
    --------
    earthkit.meteo.vertical.array.vertical_integral_pressure

    Notes
    -----
    See :func:`earthkit.meteo.vertical.array.vertical_integral_pressure` for the formulas
    and the integration schemes used for the level and the interface coordinates.
    """
    data = _as_column(data, "data")

    if p is None:
        level_type = data[0].get("vertical.level_type")
        if level_type != "pressure":
            raise ValueError(
                'The pressure could not be inferred from the metadata: "data" must be on pressure '
                f'levels when "p" is not provided. Found level type "{level_type}". Provide the '
                'pressure (Pa) as a FieldList in the "p" argument.'
            )
        coord = np.asarray(pressure_from_metadata(data), dtype=float)
    else:
        coord = _profile_fieldlist_values(p, "p", data)

    res = array.vertical_integral_pressure(data.to_numpy(copy=False), coord, vertical_dim=0)

    return _integral_fieldlist(res, data[0], "kg m-2")


def vertical_integral_height(
    data: FieldList | Field,
    h: FieldList | Field | None = None,
    rho: FieldList | Field | None = None,
) -> FieldList:
    r"""Integrate data over geometric height.

    *New in version 1.4.0*

    Computes the integral of ``data`` over the vertical axis. Without ``rho``:

    .. math::

        I = \int_{z_{bot}}^{z_{top}} f \, dz

    and with ``rho`` the mass-weighted integral:

    .. math::

        I = \int_{z_{bot}}^{z_{top}} \rho f \, dz

    where :math:`f` are the data values, :math:`z` is the geometric height (m) and
    :math:`\rho` is the air density (kg m-3). All the quantities are in SI units.

    Parameters
    ----------
    data: FieldList | Field
        Data to be integrated. It must hold a single column: one field per vertical level,
        with the same parameter, base date and time, step and ensemble member, unique
        levels and a single vertical level type. The fields are sorted by the level before
        the integration.
    h: FieldList | Field | None
        Geometric height (m) defining the vertical coordinate. Either a FieldList with
        exactly one field for each level of ``data``, on the same levels, or a FieldList
        with one more field than ``data``, holding the height of the layer interfaces
        enclosing the data levels. The fields are sorted by their own level. When None
        (default) the height is taken from the level metadata of ``data``, which must be
        on "height_above_ground_level" or "height_above_mean_sea_level" levels.
    rho: FieldList | Field | None
        Air density (kg m-3) on the levels of ``data``, in a FieldList with one field for
        each level of ``data``, on the same levels. When None (default) the plain integral
        :math:`\int f dz` is computed.

    Returns
    -------
    FieldList
        A FieldList with a single field: the vertically integrated data. Its metadata is
        copied from the first field of ``data``, the vertical level type is set to
        "entire_atmosphere" (level 0) and the units are the units of ``data`` extended with
        "kg m-2" when ``rho`` is given and with "m" when it is not. They are set to "unknown"
        when the units of ``data`` are not known or when their combination is not accepted:
        the units of ``data`` are never kept.

    Raises
    ------
    ValueError
        If ``data``, ``h`` or ``rho`` does not hold a single column, if ``data`` is not on
        height levels and ``h`` is None or if the number of fields or the levels of ``h``
        or ``rho`` do not match ``data``.

    See Also
    --------
    earthkit.meteo.vertical.array.vertical_integral_height

    Notes
    -----
    See :func:`earthkit.meteo.vertical.array.vertical_integral_height` for the formulas and
    the integration schemes used for the level and the interface coordinates.
    """
    data = _as_column(data, "data")

    if h is None:
        level_type = data[0].get("vertical.level_type")
        if level_type not in _HEIGHT_LEVEL_TYPES:
            raise ValueError(
                'The height could not be inferred from the metadata: "data" must be on '
                f'{" or ".join(_HEIGHT_LEVEL_TYPES)} levels when "h" is not provided. Found '
                f'level type "{level_type}". Provide the height (m) as a FieldList in the "h" argument.'
            )
        coord = _levels(data)
    else:
        coord = _profile_fieldlist_values(h, "h", data)

    rho_values = None
    if rho is not None:
        rho_values = _profile_fieldlist_values(rho, "rho", data, interfaces=False)

    res = array.vertical_integral_height(data.to_numpy(copy=False), coord, rho=rho_values, vertical_dim=0)

    return _integral_fieldlist(res, data[0], "kg m-2" if rho is not None else "m")
