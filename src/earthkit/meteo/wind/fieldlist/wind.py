# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import Any, TypeAlias

from earthkit.data import FieldList  # type: ignore[import]
from earthkit.utils.array import array_namespace
from earthkit.utils.units import Units

from earthkit.meteo.utils.decorators import fieldlist_ufunc

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
    param_ids = {
        131: "ws",  # atmospheric wind, paramId=10
        165: "10ws",  # 10m wind, paramId=207
        228246: "100si",  # 100m wind, paramId=228249
        228239: "200si",  # 200m wind, paramId=228241
    }

    variables = {
        "u": "ws",  # atmospheric wind
        "10u": "10ws",  # 10m wind
        "100ua": "100si",  # 100m wind
        "200ua": "200si",  # 200m wind
    }

    fieldlist_ufunc_kwargs = {
        "variables": variables,
        "param_ids": param_ids,
        "default": "ws",
    }
    return fieldlist_ufunc(array.speed, u, v, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


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
    fieldlist_ufunc_kwargs = {
        "default": "wdir",
    }
    return fieldlist_ufunc(
        array.direction,
        u,
        v,
        fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs,
        convention=convention,
        to_positive=to_positive,
    )


def xy_to_polar(x: FieldList, y: FieldList, convention: str = "meteo") -> tuple[FieldList, FieldList]:
    r"""Convert wind/vector data from xy representation to polar representation.

    Parameters
    ----------
    x: FieldList
        u wind/x vector component
    y: FieldList
        v wind/y vector component (same units as ``u``)
    convention: str
        Specify how the direction/angle component of the target polar coordinate
        system is interpreted. The possible values are as follows:

        * "meteo": the direction is the meteorological wind direction (see :func:`direction` for explanation)
        * "polar": the direction is measured anti-clockwise from the x axis (East/right) to the vector


    Returns
    -------
    FieldList
        Magnitude (same units as ``u``)
    FieldList
        Direction (degrees)

    Notes
    -----
    In the target xy representation the x axis points East while the y axis points North.

    """
    return speed(x, y), direction(x, y, convention=convention)


def polar_to_xy(
    magnitude: FieldList,
    direction: FieldList,
    convention: str = "meteo",
) -> tuple[FieldList, FieldList]:
    r"""Convert wind/vector data from polar representation to xy representation.

    Parameters
    ----------
    magnitude: FieldList
        Speed/magnitude of the vector
    direction: FieldList
        Direction of the vector (degrees)
    convention: str
        Specify how ``direction`` is interpreted. The possible values are as follows:

        * "meteo": ``direction`` is the meteorological wind direction
          (see :func:`direction` for explanation)
        * "polar": ``direction`` is the angle measured anti-clockwise from the x axis
          (East/right) to the vector

    Returns
    -------
    FieldList
        X vector component (same units as ``magnitude``)
    FieldList
        Y vector component (same units as ``magnitude``)


    Notes
    -----
    In the target xy representation the x axis points East while the y axis points North.

    """
    if len(magnitude) != len(direction):
        raise ValueError("magnitude and direction must have the same number of fields")

    result_1 = []
    result_2 = []
    for m, d in zip(magnitude, direction):
        v1, v2 = array.polar_to_xy(m.values, d.values, convention=convention)
        result_1.append(m.clone(values=v1))
        result_2.append(m.clone(values=v2))
    return magnitude.from_fields(result_1), magnitude.from_fields(result_2)


def w_from_omega(omega: FieldList, t: FieldList, p: FieldList | ArrayLike | None) -> FieldList:
    r"""Compute the hydrostatic vertical velocity from pressure velocity, temperature and pressure.

    Parameters
    ----------
    omega : FieldList
        Hydrostatic pressure velocity (Pa/s)
    t : FieldList
        Temperature (K). Must have the same number of fields as ``omega``.
    p : FieldList, array-like, None
        Pressure (Pa). If a FieldList is provided, it must have the same number of fields as ``omega``.
        If an array-like is provided, it must have the same number of elements
        as the number of fields in ``omega``.
        If None, the pressure is taken from the level information of each field in ``omega``. Only isobaric
        levels are supported in this case.

    Returns
    -------
    FieldList
        Hydrostatic vertical velocity (m/s)

    Notes
    -----
    The computation is based on the following hydrostatic formula:

    .. math::

        w = - \frac{\omega  t R_{d}}{p g}

    where

        * :math:`R_{d}` is the specific gas constant for dry air (see :data:`earthkit.meteo.constants.Rd`).
        * :math:`g` is the gravitational acceleration (see :data:`earthkit.meteo.constants.g`)

    """
    if len(omega) != len(t):
        raise ValueError(f"omega and t must have the same number of fields ({len(omega)} != {len(t)})")

    if isinstance(p, FieldList):
        if len(omega) != len(p):
            raise ValueError(f"omega and p must have the same number of fields ({len(omega)} != {len(p)})")
    elif p is None:
        p = [
            (f.get("vertical.level") * ((f.get("vertical.units", Units.from_any("hPa"))).to_pint())).to("Pa").magnitude
            for f in omega
        ]
    else:
        xp = array_namespace(p)
        p = xp.asarray(p)
        if len(p.shape) == 0:
            p = [p.item()] * len(omega)
        if len(omega) != len(p):
            raise ValueError(
                "When p is array-like, it must have the same number elements "
                f"as the number of fields in omega({len(p)} != {len(omega)})"
            )

    fieldlist_ufunc_kwargs = {
        "default": "wz",
        "param_unit": "m/s",
    }

    return fieldlist_ufunc(array.w_from_omega, omega, t, p, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)


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
        lat = field.geography.latitudes()
        c = array.coriolis(lat)
        result.append(field.set({"values": c, "parameter.variable": "fc", "parameter.units": "1/s"}))

    return FieldList.from_fields(result)
