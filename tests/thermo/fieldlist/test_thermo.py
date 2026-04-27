# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import numpy as np
import pytest

from earthkit.meteo.utils.testing import NO_EKD

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})
pytestmark = pytest.mark.skipif(NO_EKD, reason="EKD is not installed")

temperatures = [[293.15, 283.15], [273.15, 280.15], [285.54, 290.15]]  # K
pressures = [100000.0, 85000.0, 50000.0]  # Pa


def _t_fieldlist():

    from earthkit.data import Field, FieldList

    fields = []
    for t, p in zip(temperatures, pressures):
        fields.append(
            Field.from_components(
                values=np.array(t),
                parameter={"variable": "t", "units": "K"},
                vertical={"level": p / 100, "level_type": "pressure"},
            )
        )

    return FieldList.from_fields(fields)


def _t_p_fieldlists():
    from earthkit.data import Field, FieldList

    t_fields = []
    for t in temperatures:
        t_fields.append(Field.from_components(values=np.array(t), parameter={"variable": "t", "units": "K"}))

    p_fields = []
    for p in pressures:
        p_fields.append(Field.from_components(values=np.array([p]), parameter={"variable": "pres", "units": "Pa"}))

    return FieldList.from_fields(t_fields), FieldList.from_fields(p_fields)


def test_xr_potential_temperature():
    from earthkit.meteo.thermo.fieldlist import potential_temperature

    t, p = _t_p_fieldlists()
    out = potential_temperature(t, p)

    numpy_out = np.array([
        [293.1500000000, 283.1500000000],
        [286.1314413346, 293.4641160164],
        [348.0715407528, 353.6911029959],
    ])
    np.testing.assert_allclose(out.to_numpy(), numpy_out)
    assert (np.array(out.get("parameter.variable")) == "pt").all()


def test_xr_potential_temperature_infer_pressure():
    from earthkit.meteo.thermo.fieldlist import potential_temperature

    t = _t_fieldlist()
    out = potential_temperature(t)

    numpy_out = np.array([
        [293.1500000000, 283.1500000000],
        [286.1314413346, 293.4641160164],
        [348.0715407528, 353.6911029959],
    ])
    np.testing.assert_allclose(out.to_numpy(), numpy_out)
    assert (np.array(out.get("parameter.variable")) == "pt").all()
