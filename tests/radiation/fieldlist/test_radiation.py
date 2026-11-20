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

pytestmark = pytest.mark.skipif(NO_EKD, reason="EKD is not installed")

DIFFUSE = [[0.0, 120.5], [310.0, 22.0]]  # W/m2
DIRECT = [[0.0, 45.25], [590.0, 8.0]]  # W/m2
TOTAL = [[0.0, 165.75], [900.0, 30.0]]  # W/m2

PARAMETERS = {
    "diffuse": {"variable": "ssrd_diffuse", "units": "W/m2"},
    "direct": {"variable": "fdir", "units": "W/m2"},
    "net_longwave": {"variable": "athb_s", "units": "W/m2"},
    "surface_temperature": {"variable": "t", "units": "K"},
}


def _make_input_fieldlist(param, values, input_type="fieldlist"):
    from earthkit.data import Field, FieldList

    param_def = PARAMETERS[param]
    vertical = {"level": 0, "level_type": "surface"}

    fields = [
        Field.from_components(
            values=np.array(v),
            parameter={"variable": param_def["variable"], "units": param_def["units"]},
            vertical=vertical,
        )
        for v in values
    ]

    return fields[0] if input_type == "field" else FieldList.from_fields(fields)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_downward_shortwave_radiation(input_type):
    from earthkit.meteo.radiation.fieldlist import surface_downward_shortwave_radiation

    diffuse = _make_input_fieldlist("diffuse", DIFFUSE, input_type=input_type)
    direct = _make_input_fieldlist("direct", DIRECT, input_type=input_type)

    out = surface_downward_shortwave_radiation(diffuse, direct)

    assert isinstance(out, type(diffuse))

    if input_type == "fieldlist":
        assert len(out) == len(diffuse)
        assert out.get("parameter.variable") == ["avg_sdswrf"] * len(diffuse)
        for f, ref_vals in zip(out, TOTAL):
            np.testing.assert_allclose(f.values, np.array(ref_vals))
    else:
        assert out.get("parameter.variable") == "avg_sdswrf"
        np.testing.assert_allclose(out.values, np.array(TOTAL[0]))


def test_fieldlist_downward_shortwave_radiation_units():
    """The output parameter metadata is resolved from FIELD_PARAMS."""
    from earthkit.meteo.radiation.fieldlist import surface_downward_shortwave_radiation

    diffuse = _make_input_fieldlist("diffuse", DIFFUSE, input_type="field")
    direct = _make_input_fieldlist("direct", DIRECT, input_type="field")

    out = surface_downward_shortwave_radiation(diffuse, direct)

    assert str(out.get("parameter.units")) == str(diffuse.get("parameter.units"))


NET_LONGWAVE = [[-60.0, -30.0], [-105.5, -12.25]]  # W/m2
SURFACE_TEMPERATURE = [[280.0, 300.0], [265.3, 291.0]]  # K


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_surface_downward_longwave_flux(input_type):
    from earthkit.meteo.radiation import array
    from earthkit.meteo.radiation.fieldlist import surface_downward_longwave_flux

    net_longwave = _make_input_fieldlist("net_longwave", NET_LONGWAVE, input_type=input_type)
    surface_temperature = _make_input_fieldlist("surface_temperature", SURFACE_TEMPERATURE, input_type=input_type)

    out = surface_downward_longwave_flux(net_longwave, surface_temperature)

    assert isinstance(out, type(net_longwave))

    if input_type == "fieldlist":
        assert len(out) == len(net_longwave)
        assert out.get("parameter.variable") == ["avg_sdlwrf"] * len(net_longwave)
        for f, lw, t in zip(out, NET_LONGWAVE, SURFACE_TEMPERATURE):
            ref = array.surface_downward_longwave_flux(np.array(lw), np.array(t))
            np.testing.assert_allclose(f.values, ref)
    else:
        assert out.get("parameter.variable") == "avg_sdlwrf"
        ref = array.surface_downward_longwave_flux(np.array(NET_LONGWAVE[0]), np.array(SURFACE_TEMPERATURE[0]))
        np.testing.assert_allclose(out.values, ref)


def test_fieldlist_surface_downward_longwave_flux_emissivity():
    """The emissivity keyword is forwarded and not mistaken for a field argument."""
    from earthkit.meteo.radiation import array
    from earthkit.meteo.radiation.fieldlist import surface_downward_longwave_flux

    net_longwave = _make_input_fieldlist("net_longwave", NET_LONGWAVE, input_type="field")
    surface_temperature = _make_input_fieldlist("surface_temperature", SURFACE_TEMPERATURE, input_type="field")

    out = surface_downward_longwave_flux(net_longwave, surface_temperature, emissivity=0.9)
    ref = array.surface_downward_longwave_flux(
        np.array(NET_LONGWAVE[0]), np.array(SURFACE_TEMPERATURE[0]), emissivity=0.9
    )

    np.testing.assert_allclose(out.values, ref)
