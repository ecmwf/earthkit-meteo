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

TEMPERATURES = [[293.15, 283.15], [273.15, 280.15], [285.54, 290.15]]  # K
DEWPOINTS = [[283.15, 275.15], [265.15, 270.15], [275.54, 280.15]]  # K
SPECIFIC_HUMIDITIES = [[0.008, 0.005], [0.003, 0.004], [0.006, 0.007]]  # kg/kg
MIXING_RATIOS = [[0.008064, 0.005025], [0.003009, 0.004016], [0.006036, 0.007049]]  # kg/kg
RELATIVE_HUMIDITIES = [[60.0, 70.0], [50.0, 55.0], [65.0, 75.0]]  # %
PRESSURES = [100000.0, 85000.0, 50000.0]  # Pa
POTENTIAL_TEMPERATURES = [
    [293.15, 283.15],
    [286.1314413346, 293.4641160164],
    [348.0715407528, 353.6911029959],
]  # K

PARAMETERS = {
    "t": {"variable": "t", "units": "K"},
    "td": {"variable": "td", "units": "K"},
    "q": {"variable": "q", "units": "kg/kg"},
    "w": {"variable": "w", "units": "kg/kg"},
    "r": {"variable": "r", "units": "%"},
    "rh": {"variable": "rh", "units": "%"},
    "pres": {"variable": "pres", "units": "Pa"},
    "pt": {"variable": "pt", "units": "K"},
    "e": {"variable": "e", "units": "Pa"},
    "es": {"variable": "es", "units": "Pa"},
    "ept": {"variable": "ept", "units": "K"},
}


def _make_input_fieldlist(param, values, input_type="fieldlist", level_type="pressure"):
    from earthkit.data import Field, FieldList

    param_def = PARAMETERS[param]

    if input_type == "field":
        if level_type == "pressure":
            vertical = {"level": PRESSURES[0] / 100, "level_type": "pressure"}
        elif level_type == "surface":
            vertical = {"level": 0, "level_type": "surface"}
        else:
            vertical = None

        fl = Field.from_components(
            values=np.array(values[0]),
            parameter={"variable": param_def["variable"], "units": param_def["units"]},
            vertical=vertical,
        )
        return fl
    elif input_type == "fieldlist":
        if level_type == "pressure":
            fl = []
            if len(values) != len(PRESSURES):
                raise ValueError(f"Length of values ({len(values)}) must match length of pressures ({len(PRESSURES)})")

            for v, p in zip(values, PRESSURES):
                vertical = {"level": p / 100, "level_type": "pressure"}
                fl.append(
                    Field.from_components(
                        values=np.array(v),
                        parameter={"variable": param_def["variable"], "units": param_def["units"]},
                        vertical=vertical,
                    )
                )
            return FieldList.from_fields(fl)

        elif level_type == "surface":
            fl = []
            vertical = {"level": 0, "level_type": "surface"}
            fl.append(
                Field.from_components(
                    values=np.array(values[0]),
                    parameter={"variable": param_def["variable"], "units": param_def["units"]},
                    vertical=vertical,
                )
            )
            return FieldList.from_fields(fl)


def _make_pres_fieldlist(md_fl, pres_type="fl"):
    from earthkit.data import Field, FieldList

    if isinstance(md_fl, FieldList):
        if pres_type == "fl":
            p_fl = []
            for md_field in md_fl:
                p_fl.append(
                    md_field.set(
                        values=md_field.values * 0.0 + md_field.get("vertical.level") * 100.0,
                        **{"parameter.variable": "pres"},
                    ).sync()
                )
            return FieldList.from_fields(p_fl)
        elif pres_type == "value":
            pres = np.array(md_fl.get("vertical.level")) * 100.0  # convert to Pa
            return pres
    elif isinstance(md_fl, Field):
        if pres_type == "fl":
            return md_fl.set(
                values=md_fl.values * 0.0 + md_fl.get("vertical.level") * 100.0,
                **{"parameter.variable": "pres"},
            ).sync()
        elif pres_type == "value":
            pres = md_fl.get("vertical.level") * 100.0  # convert to Pa
            return pres
    else:
        raise ValueError(f"Unsupported md_fl type: {type(md_fl)}")

    if pres_type is None:
        return None

    raise ValueError(f"Unsupported pres_type: {pres_type}")


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_potential_temperature(input_type, pres_type):
    from earthkit.meteo.thermo.fieldlist import potential_temperature

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = potential_temperature(t, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["pt"] * len(t)
        assert out.get("parameter.units") == ["K"] * len(t)
        assert np.allclose(np.array(out.get("vertical.level")), np.array(PRESSURES) / 100.0)
        for f, ref_vals in zip(out, POTENTIAL_TEMPERATURES):
            np.testing.assert_allclose(f.values, np.array(ref_vals))
    elif input_type == "field":
        assert out.get("parameter.variable") == "pt"
        assert out.get("parameter.units") == "K"
        assert np.allclose(out.get("vertical.level"), PRESSURES[0] / 100.0)
        np.testing.assert_allclose(out.values, np.array(POTENTIAL_TEMPERATURES[0]))


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_specific_humidity_from_mixing_ratio(input_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import specific_humidity_from_mixing_ratio

    w = _make_input_fieldlist("w", values=MIXING_RATIOS, input_type=input_type)
    out = specific_humidity_from_mixing_ratio(w)

    assert isinstance(out, type(w))

    if input_type == "fieldlist":
        assert len(out) == len(w)
        assert out.get("parameter.variable") == ["q"] * len(w)
        for f, vals in zip(out, MIXING_RATIOS):
            ref = array.specific_humidity_from_mixing_ratio(np.array(vals))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "q"
        ref = array.specific_humidity_from_mixing_ratio(np.array(MIXING_RATIOS[0]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_mixing_ratio_from_specific_humidity(input_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import mixing_ratio_from_specific_humidity

    q = _make_input_fieldlist("q", values=SPECIFIC_HUMIDITIES, input_type=input_type)
    out = mixing_ratio_from_specific_humidity(q)

    assert isinstance(out, type(q))

    if input_type == "fieldlist":
        assert len(out) == len(q)
        assert out.get("parameter.variable") == ["mass_mixrat"] * len(q)
        for f, vals in zip(out, SPECIFIC_HUMIDITIES):
            ref = array.mixing_ratio_from_specific_humidity(np.array(vals))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "mass_mixrat"
        ref = array.mixing_ratio_from_specific_humidity(np.array(SPECIFIC_HUMIDITIES[0]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_vapour_pressure_from_specific_humidity(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import vapour_pressure_from_specific_humidity

    q = _make_input_fieldlist("q", values=SPECIFIC_HUMIDITIES, input_type=input_type)
    p = _make_pres_fieldlist(q, pres_type=pres_type)
    out = vapour_pressure_from_specific_humidity(q, p)

    assert isinstance(out, type(q))

    if input_type == "fieldlist":
        assert len(out) == len(q)
        assert out.get("parameter.variable") == ["vapp"] * len(q)
        assert out.get("parameter.units") == ["Pa"] * len(q)
        for f, vals, p_val in zip(out, SPECIFIC_HUMIDITIES, PRESSURES):
            ref = array.vapour_pressure_from_specific_humidity(np.array(vals), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "vapp"
        assert out.get("parameter.units") == "Pa"
        ref = array.vapour_pressure_from_specific_humidity(np.array(SPECIFIC_HUMIDITIES[0]), np.array([PRESSURES[0]]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_vapour_pressure_from_mixing_ratio(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import vapour_pressure_from_mixing_ratio

    w = _make_input_fieldlist("w", values=MIXING_RATIOS, input_type=input_type)
    p = _make_pres_fieldlist(w, pres_type=pres_type)
    out = vapour_pressure_from_mixing_ratio(w, p)

    assert isinstance(out, type(w))

    if input_type == "fieldlist":
        assert len(out) == len(w)
        assert out.get("parameter.variable") == ["vapp"] * len(w)
        assert out.get("parameter.units") == ["Pa"] * len(w)
        for f, vals, p_val in zip(out, MIXING_RATIOS, PRESSURES):
            ref = array.vapour_pressure_from_mixing_ratio(np.array(vals), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "vapp"
        assert out.get("parameter.units") == "Pa"
        ref = array.vapour_pressure_from_mixing_ratio(np.array(MIXING_RATIOS[0]), np.array([PRESSURES[0]]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_specific_humidity_from_vapour_pressure(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import specific_humidity_from_vapour_pressure

    e_values = [[800.0, 600.0], [500.0, 550.0], [400.0, 450.0]]
    e = _make_input_fieldlist("e", values=e_values, input_type=input_type)
    p = _make_pres_fieldlist(e, pres_type=pres_type)
    out = specific_humidity_from_vapour_pressure(e, p)

    assert isinstance(out, type(e))

    if input_type == "fieldlist":
        assert len(out) == len(e)
        assert out.get("parameter.variable") == ["q"] * len(e)
        assert out.get("parameter.units") == ["kg/kg"] * len(e)
        for f, vals, p_val in zip(out, e_values, PRESSURES):
            ref = array.specific_humidity_from_vapour_pressure(np.array(vals), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "q"
        assert out.get("parameter.units") == "kg/kg"
        ref = array.specific_humidity_from_vapour_pressure(np.array(e_values[0]), np.array([PRESSURES[0]]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_mixing_ratio_from_vapour_pressure(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import mixing_ratio_from_vapour_pressure

    e_values = [[800.0, 600.0], [500.0, 550.0], [400.0, 450.0]]
    e = _make_input_fieldlist("e", values=e_values, input_type=input_type)
    p = _make_pres_fieldlist(e, pres_type=pres_type)
    out = mixing_ratio_from_vapour_pressure(e, p)

    assert isinstance(out, type(e))

    if input_type == "fieldlist":
        assert len(out) == len(e)
        assert out.get("parameter.variable") == ["mass_mixrat"] * len(e)
        assert out.get("parameter.units") == ["kg/kg"] * len(e)
        for f, vals, p_val in zip(out, e_values, PRESSURES):
            ref = array.mixing_ratio_from_vapour_pressure(np.array(vals), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "mass_mixrat"
        assert out.get("parameter.units") == "kg/kg"
        ref = array.mixing_ratio_from_vapour_pressure(np.array(e_values[0]), np.array([PRESSURES[0]]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_saturation_vapour_pressure(input_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_vapour_pressure

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    out = saturation_vapour_pressure(t)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["swvp"] * len(t)
        assert out.get("parameter.units") == ["Pa"] * len(t)
        for f, vals in zip(out, TEMPERATURES):
            ref = array.saturation_vapour_pressure(np.array(vals))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "swvp"
        assert out.get("parameter.units") == "Pa"
        ref = array.saturation_vapour_pressure(np.array(TEMPERATURES[0]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_saturation_vapour_pressure_phase(input_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_vapour_pressure

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    out = saturation_vapour_pressure(t, phase="water")

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["swvp"] * len(t)
        assert out.get("parameter.units") == ["Pa"] * len(t)
        for f, vals in zip(out, TEMPERATURES):
            ref = array.saturation_vapour_pressure(np.array(vals), phase="water")
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "swvp"
        assert out.get("parameter.units") == "Pa"
        ref = array.saturation_vapour_pressure(np.array(TEMPERATURES[0]), phase="water")
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_saturation_mixing_ratio(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_mixing_ratio

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = saturation_mixing_ratio(t, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["ws"] * len(t)
        assert out.get("parameter.units") == ["kg/kg"] * len(t)
        for f, vals, p_val in zip(out, TEMPERATURES, PRESSURES):
            ref = array.saturation_mixing_ratio(np.array(vals), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "ws"
        assert out.get("parameter.units") == "kg/kg"
        ref = array.saturation_mixing_ratio(np.array(TEMPERATURES[0]), np.array([PRESSURES[0]]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_saturation_specific_humidity(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_specific_humidity

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = saturation_specific_humidity(t, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["sqw"] * len(t)
        assert out.get("parameter.units") == ["kg/kg"] * len(t)
        for f, vals, p_val in zip(out, TEMPERATURES, PRESSURES):
            ref = array.saturation_specific_humidity(np.array(vals), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "sqw"
        assert out.get("parameter.units") == "kg/kg"
        ref = array.saturation_specific_humidity(np.array(TEMPERATURES[0]), np.array([PRESSURES[0]]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_saturation_vapour_pressure_slope(input_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_vapour_pressure_slope

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    out = saturation_vapour_pressure_slope(t)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["swvp_slope"] * len(t)
        for f, vals in zip(out, TEMPERATURES):
            ref = array.saturation_vapour_pressure_slope(np.array(vals))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "swvp_slope"
        ref = array.saturation_vapour_pressure_slope(np.array(TEMPERATURES[0]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_saturation_mixing_ratio_slope(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_mixing_ratio_slope

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = saturation_mixing_ratio_slope(t, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["ws_slope"] * len(t)
        for f, vals, p_val in zip(out, TEMPERATURES, PRESSURES):
            ref = array.saturation_mixing_ratio_slope(np.array(vals), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "ws_slope"
        ref = array.saturation_mixing_ratio_slope(np.array(TEMPERATURES[0]), np.array([PRESSURES[0]]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_saturation_specific_humidity_slope(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_specific_humidity_slope

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = saturation_specific_humidity_slope(t, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["sqw_slope"] * len(t)
        for f, vals, p_val in zip(out, TEMPERATURES, PRESSURES):
            ref = array.saturation_specific_humidity_slope(np.array(vals), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "sqw_slope"
        ref = array.saturation_specific_humidity_slope(np.array(TEMPERATURES[0]), np.array([PRESSURES[0]]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_temperature_from_saturation_vapour_pressure(input_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import temperature_from_saturation_vapour_pressure

    es_values = [[2339.0, 1228.0], [611.0, 1000.0], [1500.0, 2000.0]]
    es = _make_input_fieldlist("es", values=es_values, input_type=input_type)
    out = temperature_from_saturation_vapour_pressure(es)

    assert isinstance(out, type(es))

    if input_type == "fieldlist":
        assert len(out) == len(es)
        assert out.get("parameter.variable") == ["t"] * len(es)
        assert out.get("parameter.units") == ["K"] * len(es)
        for f, vals in zip(out, es_values):
            ref = array.temperature_from_saturation_vapour_pressure(np.array(vals))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "t"
        assert out.get("parameter.units") == "K"
        ref = array.temperature_from_saturation_vapour_pressure(np.array(es_values[0]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_relative_humidity_from_dewpoint(input_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import relative_humidity_from_dewpoint

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    td = _make_input_fieldlist("td", values=DEWPOINTS, input_type=input_type)
    out = relative_humidity_from_dewpoint(t, td)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["r"] * len(t)
        assert out.get("parameter.units") == ["%"] * len(t)
        for f, v1, v2 in zip(out, TEMPERATURES, DEWPOINTS):
            ref = array.relative_humidity_from_dewpoint(np.array(v1), np.array(v2))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "r"
        assert out.get("parameter.units") == "%"
        ref = array.relative_humidity_from_dewpoint(np.array(TEMPERATURES[0]), np.array(DEWPOINTS[0]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_relative_humidity_from_specific_humidity(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import relative_humidity_from_specific_humidity

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    q = _make_input_fieldlist("q", values=SPECIFIC_HUMIDITIES, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = relative_humidity_from_specific_humidity(t, q, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["r"] * len(t)
        assert out.get("parameter.units") == ["%"] * len(t)
        for f, v1, v2, p_val in zip(out, TEMPERATURES, SPECIFIC_HUMIDITIES, PRESSURES):
            ref = array.relative_humidity_from_specific_humidity(np.array(v1), np.array(v2), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "r"
        assert out.get("parameter.units") == "%"
        ref = array.relative_humidity_from_specific_humidity(
            np.array(TEMPERATURES[0]), np.array(SPECIFIC_HUMIDITIES[0]), np.array([PRESSURES[0]])
        )
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_specific_humidity_from_dewpoint(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import specific_humidity_from_dewpoint

    td = _make_input_fieldlist("td", values=DEWPOINTS, input_type=input_type)
    p = _make_pres_fieldlist(td, pres_type=pres_type)
    out = specific_humidity_from_dewpoint(td, p)

    assert isinstance(out, type(td))

    if input_type == "fieldlist":
        assert len(out) == len(td)
        assert out.get("parameter.variable") == ["q"] * len(td)
        assert out.get("parameter.units") == ["kg/kg"] * len(td)
        for f, vals, p_val in zip(out, DEWPOINTS, PRESSURES):
            ref = array.specific_humidity_from_dewpoint(np.array(vals), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "q"
        assert out.get("parameter.units") == "kg/kg"
        ref = array.specific_humidity_from_dewpoint(np.array(DEWPOINTS[0]), np.array([PRESSURES[0]]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_mixing_ratio_from_dewpoint(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import mixing_ratio_from_dewpoint

    td = _make_input_fieldlist("td", values=DEWPOINTS, input_type=input_type)
    p = _make_pres_fieldlist(td, pres_type=pres_type)
    out = mixing_ratio_from_dewpoint(td, p)

    assert isinstance(out, type(td))

    if input_type == "fieldlist":
        assert len(out) == len(td)
        assert out.get("parameter.variable") == ["mass_mixrat"] * len(td)
        assert out.get("parameter.units") == ["kg/kg"] * len(td)
        for f, vals, p_val in zip(out, DEWPOINTS, PRESSURES):
            ref = array.mixing_ratio_from_dewpoint(np.array(vals), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "mass_mixrat"
        assert out.get("parameter.units") == "kg/kg"
        ref = array.mixing_ratio_from_dewpoint(np.array(DEWPOINTS[0]), np.array([PRESSURES[0]]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_specific_humidity_from_relative_humidity(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import specific_humidity_from_relative_humidity

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    r = _make_input_fieldlist("r", values=RELATIVE_HUMIDITIES, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = specific_humidity_from_relative_humidity(t, r, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["q"] * len(t)
        assert out.get("parameter.units") == ["kg/kg"] * len(t)
        for f, v1, v2, p_val in zip(out, TEMPERATURES, RELATIVE_HUMIDITIES, PRESSURES):
            ref = array.specific_humidity_from_relative_humidity(np.array(v1), np.array(v2), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "q"
        assert out.get("parameter.units") == "kg/kg"
        ref = array.specific_humidity_from_relative_humidity(
            np.array(TEMPERATURES[0]), np.array(RELATIVE_HUMIDITIES[0]), np.array([PRESSURES[0]])
        )
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_dewpoint_from_relative_humidity(input_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import dewpoint_from_relative_humidity

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    r = _make_input_fieldlist("r", values=RELATIVE_HUMIDITIES, input_type=input_type)
    out = dewpoint_from_relative_humidity(t, r)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["td"] * len(t)
        for f, v1, v2 in zip(out, TEMPERATURES, RELATIVE_HUMIDITIES):
            ref = array.dewpoint_from_relative_humidity(np.array(v1), np.array(v2))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "td"
        ref = array.dewpoint_from_relative_humidity(np.array(TEMPERATURES[0]), np.array(RELATIVE_HUMIDITIES[0]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_dewpoint_from_specific_humidity(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import dewpoint_from_specific_humidity

    q = _make_input_fieldlist("q", values=SPECIFIC_HUMIDITIES, input_type=input_type)
    p = _make_pres_fieldlist(q, pres_type=pres_type)
    out = dewpoint_from_specific_humidity(q, p)

    assert isinstance(out, type(q))

    if input_type == "fieldlist":
        assert len(out) == len(q)
        assert out.get("parameter.variable") == ["td"] * len(q)
        assert out.get("parameter.units") == ["K"] * len(q)
        for f, vals, p_val in zip(out, SPECIFIC_HUMIDITIES, PRESSURES):
            ref = array.dewpoint_from_specific_humidity(np.array(vals), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "td"
        assert out.get("parameter.units") == "K"
        ref = array.dewpoint_from_specific_humidity(np.array(SPECIFIC_HUMIDITIES[0]), np.array([PRESSURES[0]]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_virtual_temperature(input_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import virtual_temperature

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    q = _make_input_fieldlist("q", values=SPECIFIC_HUMIDITIES, input_type=input_type)
    out = virtual_temperature(t, q)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["vtmp"] * len(t)
        for f, v1, v2 in zip(out, TEMPERATURES, SPECIFIC_HUMIDITIES):
            ref = array.virtual_temperature(np.array(v1), np.array(v2))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "vtmp"
        ref = array.virtual_temperature(np.array(TEMPERATURES[0]), np.array(SPECIFIC_HUMIDITIES[0]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_virtual_potential_temperature(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import virtual_potential_temperature

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    q = _make_input_fieldlist("q", values=SPECIFIC_HUMIDITIES, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = virtual_potential_temperature(t, q, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["vptmp"] * len(t)
        for f, v1, v2, p_val in zip(out, TEMPERATURES, SPECIFIC_HUMIDITIES, PRESSURES):
            ref = array.virtual_potential_temperature(np.array(v1), np.array(v2), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "vptmp"
        ref = array.virtual_potential_temperature(
            np.array(TEMPERATURES[0]), np.array(SPECIFIC_HUMIDITIES[0]), np.array([PRESSURES[0]])
        )
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_temperature_from_potential_temperature(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import temperature_from_potential_temperature

    th_values = [[300.0, 290.0], [310.0, 305.0], [350.0, 340.0]]
    th = _make_input_fieldlist("pt", values=th_values, input_type=input_type)
    p = _make_pres_fieldlist(th, pres_type=pres_type)
    out = temperature_from_potential_temperature(th, p)

    assert isinstance(out, type(th))

    if input_type == "fieldlist":
        assert len(out) == len(th)
        assert out.get("parameter.variable") == ["t"] * len(th)
        for f, vals, p_val in zip(out, th_values, PRESSURES):
            ref = array.temperature_from_potential_temperature(np.array(vals), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "t"
        ref = array.temperature_from_potential_temperature(np.array(th_values[0]), np.array([PRESSURES[0]]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_pressure_on_dry_adiabat(input_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import pressure_on_dry_adiabat

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    t_def = _make_input_fieldlist("t", values=[[300.0, 295.0]] * 3, input_type=input_type)
    p_def = _make_input_fieldlist("pres", values=[[100000.0, 100000.0]] * 3, input_type=input_type)
    out = pressure_on_dry_adiabat(t, t_def, p_def)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["pres"] * len(t)
        assert out.get("parameter.units") == ["Pa"] * len(t)
        for f, t_vals in zip(out, TEMPERATURES):
            ref = array.pressure_on_dry_adiabat(
                np.array(t_vals), np.array([300.0, 295.0]), np.array([100000.0, 100000.0])
            )
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "pres"
        assert out.get("parameter.units") == "Pa"
        ref = array.pressure_on_dry_adiabat(
            np.array(TEMPERATURES[0]), np.array([300.0, 295.0]), np.array([100000.0, 100000.0])
        )
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_temperature_on_dry_adiabat(input_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import temperature_on_dry_adiabat

    p = _make_input_fieldlist("pres", values=[[p_val] for p_val in PRESSURES], input_type=input_type)
    t_def = _make_input_fieldlist("t", values=[[300.0]] * 3, input_type=input_type)
    p_def = _make_input_fieldlist("pres", values=[[100000.0]] * 3, input_type=input_type)
    out = temperature_on_dry_adiabat(p, t_def, p_def)

    assert isinstance(out, type(p))

    if input_type == "fieldlist":
        assert len(out) == len(p)
        assert out.get("parameter.variable") == ["t"] * len(p)
        assert out.get("parameter.units") == ["K"] * len(p)
        for f, p_val in zip(out, PRESSURES):
            ref = array.temperature_on_dry_adiabat(np.array([p_val]), np.array([300.0]), np.array([100000.0]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "t"
        assert out.get("parameter.units") == "K"
        ref = array.temperature_on_dry_adiabat(np.array([PRESSURES[0]]), np.array([300.0]), np.array([100000.0]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_lcl_temperature(input_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import lcl_temperature

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    td = _make_input_fieldlist("td", values=DEWPOINTS, input_type=input_type)
    out = lcl_temperature(t, td)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["t_lcl"] * len(t)
        for f, v1, v2 in zip(out, TEMPERATURES, DEWPOINTS):
            ref = array.lcl_temperature(np.array(v1), np.array(v2))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "t_lcl"
        ref = array.lcl_temperature(np.array(TEMPERATURES[0]), np.array(DEWPOINTS[0]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_lcl(input_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import lcl

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    td = _make_input_fieldlist("td", values=DEWPOINTS, input_type=input_type)
    p = _make_input_fieldlist("pres", values=[[p_val, p_val] for p_val in PRESSURES], input_type=input_type)
    t_out, p_out = lcl(t, td, p)

    assert isinstance(t_out, type(t))

    if input_type == "fieldlist":
        assert len(t_out) == len(t)
        assert t_out.get("parameter.variable") == ["t_lcl"] * len(t)
        assert p_out.get("parameter.variable") == ["p_lcl"] * len(t)
        assert t_out.get("parameter.units") == ["K"] * len(t)
        assert p_out.get("parameter.units") == ["Pa"] * len(t)
        for tf, pf, t_vals, td_vals, p_val in zip(t_out, p_out, TEMPERATURES, DEWPOINTS, PRESSURES):
            t_ref, p_ref = array.lcl(np.array(t_vals), np.array(td_vals), np.array([p_val, p_val]))
            np.testing.assert_allclose(tf.values, t_ref)
            np.testing.assert_allclose(pf.values, p_ref)
    elif input_type == "field":
        assert t_out.get("parameter.variable") == "t_lcl"
        assert p_out.get("parameter.variable") == "p_lcl"
        assert t_out.get("parameter.units") == "K"
        assert p_out.get("parameter.units") == "Pa"
        t_ref, p_ref = array.lcl(
            np.array(TEMPERATURES[0]), np.array(DEWPOINTS[0]), np.array([PRESSURES[0], PRESSURES[0]])
        )
        np.testing.assert_allclose(t_out.values, t_ref)
        np.testing.assert_allclose(p_out.values, p_ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_ept_from_dewpoint(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import ept_from_dewpoint

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    td = _make_input_fieldlist("td", values=DEWPOINTS, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = ept_from_dewpoint(t, td, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["eqpt"] * len(t)
        for f, v1, v2, p_val in zip(out, TEMPERATURES, DEWPOINTS, PRESSURES):
            ref = array.ept_from_dewpoint(np.array(v1), np.array(v2), np.array([p_val, p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "eqpt"
        ref = array.ept_from_dewpoint(
            np.array(TEMPERATURES[0]), np.array(DEWPOINTS[0]), np.array([PRESSURES[0], PRESSURES[0]])
        )
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_ept_from_specific_humidity(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import ept_from_specific_humidity

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    q = _make_input_fieldlist("q", values=SPECIFIC_HUMIDITIES, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = ept_from_specific_humidity(t, q, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["eqpt"] * len(t)
        for f, v1, v2, p_val in zip(out, TEMPERATURES, SPECIFIC_HUMIDITIES, PRESSURES):
            ref = array.ept_from_specific_humidity(np.array(v1), np.array(v2), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "eqpt"
        ref = array.ept_from_specific_humidity(
            np.array(TEMPERATURES[0]), np.array(SPECIFIC_HUMIDITIES[0]), np.array([PRESSURES[0]])
        )
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_saturation_ept(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_ept

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = saturation_ept(t, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["sept"] * len(t)
        for f, vals, p_val in zip(out, TEMPERATURES, PRESSURES):
            ref = array.saturation_ept(np.array(vals), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "sept"
        ref = array.saturation_ept(np.array(TEMPERATURES[0]), np.array([PRESSURES[0]]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_temperature_on_moist_adiabat(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import temperature_on_moist_adiabat

    ept_values = [[330.0, 320.0], [340.0, 335.0], [350.0, 345.0]]
    ept = _make_input_fieldlist("ept", values=ept_values, input_type=input_type)
    p = _make_pres_fieldlist(ept, pres_type=pres_type)
    out = temperature_on_moist_adiabat(ept, p)

    assert isinstance(out, type(ept))

    if input_type == "fieldlist":
        assert len(out) == len(ept)
        assert out.get("parameter.variable") == ["t"] * len(ept)
        for f, vals, p_val in zip(out, ept_values, PRESSURES):
            ref = array.temperature_on_moist_adiabat(np.array(vals), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "t"
        ref = array.temperature_on_moist_adiabat(np.array(ept_values[0]), np.array([PRESSURES[0]]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_wet_bulb_temperature_from_dewpoint(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import wet_bulb_temperature_from_dewpoint

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    td = _make_input_fieldlist("td", values=DEWPOINTS, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = wet_bulb_temperature_from_dewpoint(t, td, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["wbgt"] * len(t)
        for f, v1, v2, p_val in zip(out, TEMPERATURES, DEWPOINTS, PRESSURES):
            ref = array.wet_bulb_temperature_from_dewpoint(np.array(v1), np.array(v2), np.array([p_val, p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "wbgt"
        ref = array.wet_bulb_temperature_from_dewpoint(
            np.array(TEMPERATURES[0]), np.array(DEWPOINTS[0]), np.array([PRESSURES[0], PRESSURES[0]])
        )
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_wet_bulb_temperature_from_specific_humidity(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import wet_bulb_temperature_from_specific_humidity

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    q = _make_input_fieldlist("q", values=SPECIFIC_HUMIDITIES, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = wet_bulb_temperature_from_specific_humidity(t, q, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["wbgt"] * len(t)
        for f, v1, v2, p_val in zip(out, TEMPERATURES, SPECIFIC_HUMIDITIES, PRESSURES):
            ref = array.wet_bulb_temperature_from_specific_humidity(np.array(v1), np.array(v2), np.array([p_val]))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "wbgt"
        ref = array.wet_bulb_temperature_from_specific_humidity(
            np.array(TEMPERATURES[0]), np.array(SPECIFIC_HUMIDITIES[0]), np.array([PRESSURES[0]])
        )
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_wet_bulb_potential_temperature_from_dewpoint(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import wet_bulb_potential_temperature_from_dewpoint

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    td = _make_input_fieldlist("td", values=DEWPOINTS, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = wet_bulb_potential_temperature_from_dewpoint(t, td, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["wbpt"] * len(t)
        for f, v1, v2, p_val in zip(out, TEMPERATURES, DEWPOINTS, PRESSURES):
            ref = array.wet_bulb_potential_temperature_from_dewpoint(
                np.array(v1), np.array(v2), np.array([p_val, p_val])
            )
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "wbpt"
        ref = array.wet_bulb_potential_temperature_from_dewpoint(
            np.array(TEMPERATURES[0]), np.array(DEWPOINTS[0]), np.array([PRESSURES[0], PRESSURES[0]])
        )
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_wet_bulb_potential_temperature_from_specific_humidity(input_type, pres_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import wet_bulb_potential_temperature_from_specific_humidity

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    q = _make_input_fieldlist("q", values=SPECIFIC_HUMIDITIES, input_type=input_type)
    p = _make_pres_fieldlist(t, pres_type=pres_type)
    out = wet_bulb_potential_temperature_from_specific_humidity(t, q, p)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["wbpt"] * len(t)
        for f, v1, v2, p_val in zip(out, TEMPERATURES, SPECIFIC_HUMIDITIES, PRESSURES):
            ref = array.wet_bulb_potential_temperature_from_specific_humidity(
                np.array(v1), np.array(v2), np.array([p_val])
            )
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "wbpt"
        ref = array.wet_bulb_potential_temperature_from_specific_humidity(
            np.array(TEMPERATURES[0]), np.array(SPECIFIC_HUMIDITIES[0]), np.array([PRESSURES[0]])
        )
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_specific_gas_constant(input_type):
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import specific_gas_constant

    q = _make_input_fieldlist("q", values=SPECIFIC_HUMIDITIES, input_type=input_type)
    out = specific_gas_constant(q)

    assert isinstance(out, type(q))

    if input_type == "fieldlist":
        assert len(out) == len(q)
        assert out.get("parameter.variable") == ["R"] * len(q)
        assert out.get("parameter.units") == ["J kg-1 K-1"] * len(q)
        for f, vals in zip(out, SPECIFIC_HUMIDITIES):
            ref = array.specific_gas_constant(np.array(vals))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "R"
        assert out.get("parameter.units") == "J kg-1 K-1"
        ref = array.specific_gas_constant(np.array(SPECIFIC_HUMIDITIES[0]))
        np.testing.assert_allclose(out.values, ref)
