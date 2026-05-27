# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import Any

from earthkit.data import Field, FieldList  # type: ignore[import]
from numpy.typing import NDArray


def field_pressure_in_pa(field: Field) -> float:
    from earthkit.utils.units import Units

    level = field.get("vertical.level")
    unit = field.get("vertical.units", Units.from_any("hPa"))
    return (level * unit.to_pint()).to("Pa").magnitude


def pressure_from_metadata(fields: Field | FieldList) -> float | list[float]:
    """Infer pressure in Pa from field metadata.

    Parameters
    ----------
    fields: Field or FieldList
        The field(s) from which the pressure will be inferred from the metadata.

    Returns
    -------
    float or list of float
        The inferred pressure in Pa. If a single Field is provided, a single float is returned.
        If a FieldList is provided, a list of floats is returned.
    """
    import earthkit.data as ekd

    if isinstance(fields, ekd.Field):
        return field_pressure_in_pa(fields)

    return [field_pressure_in_pa(f) for f in fields]


def surface_pressure_values(field, copy=False):
    import numpy as np

    arr = field.to_numpy(copy=copy)
    first_value = arr.flat[np.argmax(arr == arr)]

    # If the first value is less than 20.0, we assume that this is the logarithm of the surface
    #  pressure, and we exponentiate it to get the actual surface pressure in Pa.
    if first_value < 20.0:
        return np.exp(arr)
    else:
        return arr


def hybrid_level_parameters_from_fieldlist(*args) -> tuple[NDArray[Any], NDArray[Any]]:
    import earthkit.data as ekd
    import numpy as np

    for fl in args:
        if isinstance(fl, ekd.Field):
            fl = [fl]
        elif not isinstance(fl, ekd.FieldList):
            continue

        for field in fl:
            if field.get("vertical.level_type") == "hybrid":
                pv_num = field.get("metadata.NV")
                try:
                    if pv_num is not None and pv_num > 2:
                        pv_num = int(pv_num)
                        coeff_num = int(pv_num / 2)
                        pv = field.get("metadata.pv")

                        pv = field.get("metadata.pv")
                        if pv is not None and len(pv) == pv_num:
                            A = np.array(pv[:coeff_num])
                            B = np.array(pv[coeff_num:])
                            return A, B
                except Exception:
                    # print("Error while inferring hybrid level parameters:", e)
                    pass
    return None, None


def get_hybrid_level_parameters(*args, A=None, B=None):
    if A is not None and B is None:
        raise ValueError("When A is provided, B must also be provided.")
    if A is None and B is not None:
        raise ValueError("When B is provided, A must also be provided.")
    if A is None or B is None:
        A, B = hybrid_level_parameters_from_fieldlist(*args)
        if A is None or B is None:
            raise ValueError("A and B parameters could not be inferred from the input fields.")

    assert A is not None and B is not None, "A and B parameters must be provided or inferred from the input fields."

    if len(A) != len(B):
        raise ValueError(f"A and B must have the same length. Got len(A)={len(A)}, len(B)={len(B)}")

    return A, B
