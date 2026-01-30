# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from typing import Any
from typing import TypeAlias

from earthkit.data import FieldList  # type: ignore[import]
from earthkit.utils.array import array_namespace

from .. import array

ArrayLike: TypeAlias = Any


def speed(u: FieldList, v: FieldList) -> FieldList:
    r"""Compute the wind speed/vector magnitude.

    Parameters
    ----------
    u: FieldList
        u wind/x vector component
    v: FieldList
        v wind/y vector component (same units as ``u``)

    Returns
    -------
    FieldList
        Wind speed/magnitude (same units as ``u`` and ``v``)
    """
    if len(u) != len(v):
        raise ValueError("u and v must have the same number of fields")

    # Mapping u-wind component GRIB parameter IDs to wind speed parameter IDs
    param_ids = {
        131: 10,  # atmospheric wind
        165: 207,  # 10m wind
        228246: 228249,  # 100m wind
        228239: 228241,  # 200m wind
    }

    result = []
    for ui, vi in zip(u, v):
        v = array.speed(ui.values, vi.values)

        param_id_u = ui.metadata("paramId", default=None)
        param_id_sp = param_ids.get(param_id_u, 10)
        keys = {}
        if param_id_sp is not None:
            keys["paramId"] = param_id_sp

        md = ui.metadata().override(**keys)
        result.append(ui.clone(values=v, metadata=md))
    return u.from_fields(result)


def direction(u: FieldList, v: FieldList, convention="meteo", to_positive=True) -> FieldList:
    r"""Compute the direction/angle of a vector quantity.

    Parameters
    ----------
    u: FieldList
        u wind/x vector component
    v: FieldList
        v wind/y vector component (same units as ``u``)
    convention: str, optional
        Specify how the direction/angle is interpreted. The possible values are as follows:

        * "meteo": the direction is the meteorological wind direction (see below for explanation)
        * "polar": the direction is measured anti-clockwise from the x axis (East/right) to the vector

    to_positive: bool, optional
        If True, the resulting values are mapped into the [0, 360] range when
        ``convention`` is "polar". Otherwise they lie in the [-180, 180] range.


    Returns
    -------
    FieldList
        Direction/angle (degrees)


    Notes
    -----
    The meteorological wind direction is the direction from which the wind is
    blowing. Wind direction increases clockwise such that a northerly wind
    is 0°, an easterly wind is 90°, a southerly wind is 180°, and a westerly
    wind is 270°. The figure below illustrates how it is related to the actual
    orientation of the wind vector:

    .. image:: /_static/wind_direction.png
        :width: 400px

    """
    if len(u) != len(v):
        raise ValueError("u and v must have the same number of fields")

    wind_param_ids = {131, 165, 228246, 228239}
    dir_param_id = 3031  # wind direction

    result = []
    for ui, vi in zip(u, v):
        v = array.direction(ui.values, vi.values)

        param_id_u = ui.metadata("paramId", default=None)
        keys = {}
        if param_id_u in wind_param_ids:
            keys["paramId"] = dir_param_id
        md = ui.metadata().override(**keys)

        result.append(ui.clone(values=v, metadata=md))
    return u.from_fields(result)


def w_from_omega(omega: FieldList, t: FieldList, p: FieldList | ArrayLike) -> FieldList:
    r"""Compute the hydrostatic vertical velocity from pressure velocity, temperature and pressure.

    Parameters
    ----------
    omega : FieldList
        Hydrostatic pressure velocity (Pa/s)
    t : FieldList
        Temperature (K). Must have the same number of fields as ``omega``.
    p : FieldList, array-like
        Pressure (Pa)

    Returns
    -------
    FieldList
        Hydrostatic vertical velocity (m/s)

    Notes
    -----
    The computation is based on the following hydrostatic formula:

    .. math::

        w = - \frac{\omega\; t R_{d}}{p g}

    where

        * :math:`R_{d}` is the specific gas constant for dry air (see :data:`earthkit.meteo.constants.Rd`).
        * :math:`g` is the gravitational acceleration (see :data:`earthkit.meteo.constants.g`)

    """
    # TODO: ecCodes does not allow to set this id yet
    # w_param_id = 260238  #  geometric vertical velocity
    # out_md_keys = {"paramId": w_param_id}
    out_md_keys = {}
    if len(omega) != len(t):
        raise ValueError(f"omega and t must have the same number of fields ({len(omega)} != {len(t)})")

    if isinstance(p, FieldList):
        if len(omega) != len(p):
            raise ValueError(f"omega and p must have the same number of fields ({len(omega)} != {len(p)})")
    elif p is None:
        p = [None] * len(omega)
    else:
        xp = array_namespace(p)
        p = xp.asarray(p)
        if len(p.shape) == 0:
            p = [p.item()] * len(omega)
        if len(omega) != len(p):
            raise ValueError(
                f"When p is array-like, it must have the same number elements as the number of fields in omega({len(p)} != {len(omega)})"
            )

    def _pressure(field, p_input=None):
        if p_input is None:
            level, level_type = field.metadata("level", "typeOfLevel")
            if level_type == "isobaricInhPa":
                p_value = level * 100.0  # hPa to Pa
            elif level_type == "isobaricInPa":
                p_value = level
            else:
                raise ValueError(
                    f"Pressure level type '{level_type}' not supported. "
                    "Only isobaric levels are supported when p is not provided."
                )
            return p_value
        elif isinstance(p, FieldList):
            # p_input is a Field
            return p_input.values
        else:
            return p_input

    result = []
    for oi, ti, pi in zip(omega, t, p):
        p_value = _pressure(oi, pi)
        v = array.w_from_omega(oi.values, ti.values, p_value)
        md = oi.metadata().override(**out_md_keys)
        result.append(oi.clone(values=v, metadata=md))

    return omega.from_fields(result)


def coriolis(data: FieldList) -> FieldList:
    r"""Compute the Coriolis parameter.

    Parameters
    ----------
    data : FieldList
        FieldList for which to compute the Coriolis parameter. The
        latitude values are taken from the latitude/longitude representation of each field.

    Returns
    -------
    FieldList
        The Coriolis parameter (:math:`s^{-1}`)

    Notes
    -----
    The Coriolis parameter is defined by the following formula:

    .. math::

        f = 2 \Omega sin(\phi)

    where :math:`\Omega` is the rotation rate of Earth
    (see :data:`earthkit.meteo.constants.omega`) and :math:`\phi` is the latitude.

    """
    result = []
    for field in data:
        lat = field.to_latlon()["lat"]
        c = array.coriolis(lat.values)
        md = field.metadata().override(paramId=500235)
        result.append(field.clone(values=c, metadata=md))

    return data.from_fields(result)
