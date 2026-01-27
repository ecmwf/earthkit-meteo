# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from .. import array


def speed(u, v):
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
