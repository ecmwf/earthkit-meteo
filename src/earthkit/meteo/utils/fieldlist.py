# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from earthkit.data import Field, FieldList  # type: ignore[import]


def field_pressure_in_pa(field: "Field") -> float:
    from earthkit.utils.units import Units

    level = field.get("vertical.level")
    unit = field.get("vertical.units", Units.from_any("hPa"))
    return (level * unit.to_pint()).to("Pa").magnitude


def pressure_from_metadata(fields: "Field | FieldList") -> float | list[float]:
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
