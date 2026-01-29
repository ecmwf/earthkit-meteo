# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.data import FieldList  # type: ignore[import]

from .. import array


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
        if param_id_u not in wind_param_ids:
            keys["paramId"] = dir_param_id
        md = ui.metadata().override(**keys)

        result.append(ui.clone(values=v, metadata=md))
    return u.from_fields(result)
