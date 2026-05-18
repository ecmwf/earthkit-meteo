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
dewpoints = [[283.15, 275.15], [265.15, 270.15], [275.54, 280.15]]  # K
specific_humidities = [[0.008, 0.005], [0.003, 0.004], [0.006, 0.007]]  # kg/kg
mixing_ratios = [[0.008064, 0.005025], [0.003009, 0.004016], [0.006036, 0.007049]]  # kg/kg
relative_humidities = [[60.0, 70.0], [50.0, 55.0], [65.0, 75.0]]  # %
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


def _make_fieldlist(values_list, variable="x", units="1"):
    """Create a FieldList from a list of value arrays."""
    from earthkit.data import Field, FieldList

    fields = []
    for vals in values_list:
        fields.append(
            Field.from_components(
                values=np.array(vals),
                parameter={"variable": variable, "units": units},
            )
        )
    return FieldList.from_fields(fields)


def _make_fieldlist_with_pressure(values_list, variable="x", units="1"):
    """Create a FieldList from a list of value arrays with pressure level metadata."""
    from earthkit.data import Field, FieldList

    fields = []
    for vals, p in zip(values_list, pressures):
        fields.append(
            Field.from_components(
                values=np.array(vals),
                parameter={"variable": variable, "units": units},
                vertical={"level": p / 100, "level_type": "pressure"},
            )
        )
    return FieldList.from_fields(fields)


def test_fieldlist_potential_temperature():
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


def test_fieldlist_potential_temperature_infer_pressure():
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


def test_fieldlist_specific_humidity_from_mixing_ratio():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import specific_humidity_from_mixing_ratio

    w = _make_fieldlist(mixing_ratios, variable="w", units="kg/kg")
    out = specific_humidity_from_mixing_ratio(w)

    for i, vals in enumerate(mixing_ratios):
        ref = array.specific_humidity_from_mixing_ratio(np.array(vals))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "q").all()


def test_fieldlist_mixing_ratio_from_specific_humidity():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import mixing_ratio_from_specific_humidity

    q = _make_fieldlist(specific_humidities, variable="q", units="kg/kg")
    out = mixing_ratio_from_specific_humidity(q)

    for i, vals in enumerate(specific_humidities):
        ref = array.mixing_ratio_from_specific_humidity(np.array(vals))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "w").all()


def test_fieldlist_vapour_pressure_from_specific_humidity():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import vapour_pressure_from_specific_humidity

    q = _make_fieldlist(specific_humidities, variable="q", units="kg/kg")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = vapour_pressure_from_specific_humidity(q, p)

    for i, (q_vals, p_val) in enumerate(zip(specific_humidities, pressures)):
        ref = array.vapour_pressure_from_specific_humidity(np.array(q_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "vapp").all()
    assert (np.array(out.get("parameter.units")) == "Pa").all()


def test_fieldlist_vapour_pressure_from_mixing_ratio():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import vapour_pressure_from_mixing_ratio

    w = _make_fieldlist(mixing_ratios, variable="w", units="kg/kg")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = vapour_pressure_from_mixing_ratio(w, p)

    for i, (w_vals, p_val) in enumerate(zip(mixing_ratios, pressures)):
        ref = array.vapour_pressure_from_mixing_ratio(np.array(w_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "vapp").all()
    assert (np.array(out.get("parameter.units")) == "Pa").all()


def test_fieldlist_specific_humidity_from_vapour_pressure():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import specific_humidity_from_vapour_pressure

    e_values = [[800.0, 600.0], [500.0, 550.0], [400.0, 450.0]]
    e = _make_fieldlist(e_values, variable="e", units="Pa")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = specific_humidity_from_vapour_pressure(e, p)

    for i, (e_vals, p_val) in enumerate(zip(e_values, pressures)):
        ref = array.specific_humidity_from_vapour_pressure(np.array(e_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "q").all()
    assert (np.array(out.get("parameter.units")) == "kg/kg").all()


def test_fieldlist_mixing_ratio_from_vapour_pressure():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import mixing_ratio_from_vapour_pressure

    e_values = [[800.0, 600.0], [500.0, 550.0], [400.0, 450.0]]
    e = _make_fieldlist(e_values, variable="e", units="Pa")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = mixing_ratio_from_vapour_pressure(e, p)

    for i, (e_vals, p_val) in enumerate(zip(e_values, pressures)):
        ref = array.mixing_ratio_from_vapour_pressure(np.array(e_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "w").all()
    assert (np.array(out.get("parameter.units")) == "kg/kg").all()


def test_fieldlist_saturation_vapour_pressure():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_vapour_pressure

    t = _make_fieldlist(temperatures, variable="t", units="K")
    out = saturation_vapour_pressure(t)

    for i, t_vals in enumerate(temperatures):
        ref = array.saturation_vapour_pressure(np.array(t_vals))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "swvp").all()
    assert (np.array(out.get("parameter.units")) == "Pa").all()


def test_fieldlist_saturation_vapour_pressure_phase():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_vapour_pressure

    t = _make_fieldlist(temperatures, variable="t", units="K")
    out = saturation_vapour_pressure(t, phase="water")

    for i, t_vals in enumerate(temperatures):
        ref = array.saturation_vapour_pressure(np.array(t_vals), phase="water")
        np.testing.assert_allclose(out[i].values, ref)


def test_fieldlist_saturation_mixing_ratio():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_mixing_ratio

    t = _make_fieldlist(temperatures, variable="t", units="K")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = saturation_mixing_ratio(t, p)

    for i, (t_vals, p_val) in enumerate(zip(temperatures, pressures)):
        ref = array.saturation_mixing_ratio(np.array(t_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "ws").all()
    assert (np.array(out.get("parameter.units")) == "kg/kg").all()


def test_fieldlist_saturation_specific_humidity():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_specific_humidity

    t = _make_fieldlist(temperatures, variable="t", units="K")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = saturation_specific_humidity(t, p)

    for i, (t_vals, p_val) in enumerate(zip(temperatures, pressures)):
        ref = array.saturation_specific_humidity(np.array(t_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "sqw").all()
    assert (np.array(out.get("parameter.units")) == "kg/kg").all()


def test_fieldlist_saturation_vapour_pressure_slope():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_vapour_pressure_slope

    t = _make_fieldlist(temperatures, variable="t", units="K")
    out = saturation_vapour_pressure_slope(t)

    for i, t_vals in enumerate(temperatures):
        ref = array.saturation_vapour_pressure_slope(np.array(t_vals))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "es_slope").all()


def test_fieldlist_saturation_mixing_ratio_slope():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_mixing_ratio_slope

    t = _make_fieldlist(temperatures, variable="t", units="K")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = saturation_mixing_ratio_slope(t, p)

    for i, (t_vals, p_val) in enumerate(zip(temperatures, pressures)):
        ref = array.saturation_mixing_ratio_slope(np.array(t_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "ws_slope").all()


def test_fieldlist_saturation_specific_humidity_slope():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_specific_humidity_slope

    t = _make_fieldlist(temperatures, variable="t", units="K")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = saturation_specific_humidity_slope(t, p)

    for i, (t_vals, p_val) in enumerate(zip(temperatures, pressures)):
        ref = array.saturation_specific_humidity_slope(np.array(t_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "sqw_slope").all()


def test_fieldlist_temperature_from_saturation_vapour_pressure():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import temperature_from_saturation_vapour_pressure

    es_values = [[2339.0, 1228.0], [611.0, 1000.0], [1500.0, 2000.0]]
    es = _make_fieldlist(es_values, variable="es", units="Pa")
    out = temperature_from_saturation_vapour_pressure(es)

    for i, es_vals in enumerate(es_values):
        ref = array.temperature_from_saturation_vapour_pressure(np.array(es_vals))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "t").all()
    assert (np.array(out.get("parameter.units")) == "K").all()


def test_fieldlist_relative_humidity_from_dewpoint():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import relative_humidity_from_dewpoint

    t = _make_fieldlist(temperatures, variable="t", units="K")
    td = _make_fieldlist(dewpoints, variable="td", units="K")
    out = relative_humidity_from_dewpoint(t, td)

    for i, (t_vals, td_vals) in enumerate(zip(temperatures, dewpoints)):
        ref = array.relative_humidity_from_dewpoint(np.array(t_vals), np.array(td_vals))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "r").all()
    assert (np.array(out.get("parameter.units")) == "%").all()


def test_fieldlist_relative_humidity_from_specific_humidity():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import relative_humidity_from_specific_humidity

    t = _make_fieldlist(temperatures, variable="t", units="K")
    q = _make_fieldlist(specific_humidities, variable="q", units="kg/kg")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = relative_humidity_from_specific_humidity(t, q, p)

    for i, (t_vals, q_vals, p_val) in enumerate(zip(temperatures, specific_humidities, pressures)):
        ref = array.relative_humidity_from_specific_humidity(np.array(t_vals), np.array(q_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "r").all()
    assert (np.array(out.get("parameter.units")) == "%").all()


def test_fieldlist_specific_humidity_from_dewpoint():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import specific_humidity_from_dewpoint

    td = _make_fieldlist(dewpoints, variable="td", units="K")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = specific_humidity_from_dewpoint(td, p)

    for i, (td_vals, p_val) in enumerate(zip(dewpoints, pressures)):
        ref = array.specific_humidity_from_dewpoint(np.array(td_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "q").all()
    assert (np.array(out.get("parameter.units")) == "kg/kg").all()


def test_fieldlist_mixing_ratio_from_dewpoint():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import mixing_ratio_from_dewpoint

    td = _make_fieldlist(dewpoints, variable="td", units="K")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = mixing_ratio_from_dewpoint(td, p)

    for i, (td_vals, p_val) in enumerate(zip(dewpoints, pressures)):
        ref = array.mixing_ratio_from_dewpoint(np.array(td_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "w").all()
    assert (np.array(out.get("parameter.units")) == "kg/kg").all()


def test_fieldlist_specific_humidity_from_relative_humidity():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import specific_humidity_from_relative_humidity

    t = _make_fieldlist(temperatures, variable="t", units="K")
    r = _make_fieldlist(relative_humidities, variable="r", units="%")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = specific_humidity_from_relative_humidity(t, r, p)

    for i, (t_vals, r_vals, p_val) in enumerate(zip(temperatures, relative_humidities, pressures)):
        ref = array.specific_humidity_from_relative_humidity(np.array(t_vals), np.array(r_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "q").all()
    assert (np.array(out.get("parameter.units")) == "kg/kg").all()


def test_fieldlist_dewpoint_from_relative_humidity():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import dewpoint_from_relative_humidity

    t = _make_fieldlist(temperatures, variable="t", units="K")
    r = _make_fieldlist(relative_humidities, variable="r", units="%")
    out = dewpoint_from_relative_humidity(t, r)

    for i, (t_vals, r_vals) in enumerate(zip(temperatures, relative_humidities)):
        ref = array.dewpoint_from_relative_humidity(np.array(t_vals), np.array(r_vals))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "td").all()


def test_fieldlist_dewpoint_from_specific_humidity():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import dewpoint_from_specific_humidity

    q = _make_fieldlist(specific_humidities, variable="q", units="kg/kg")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = dewpoint_from_specific_humidity(q, p)

    for i, (q_vals, p_val) in enumerate(zip(specific_humidities, pressures)):
        ref = array.dewpoint_from_specific_humidity(np.array(q_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "td").all()
    assert (np.array(out.get("parameter.units")) == "K").all()


def test_fieldlist_virtual_temperature():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import virtual_temperature

    t = _make_fieldlist(temperatures, variable="t", units="K")
    q = _make_fieldlist(specific_humidities, variable="q", units="kg/kg")
    out = virtual_temperature(t, q)

    for i, (t_vals, q_vals) in enumerate(zip(temperatures, specific_humidities)):
        ref = array.virtual_temperature(np.array(t_vals), np.array(q_vals))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "vtmp").all()


def test_fieldlist_virtual_potential_temperature():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import virtual_potential_temperature

    t = _make_fieldlist(temperatures, variable="t", units="K")
    q = _make_fieldlist(specific_humidities, variable="q", units="kg/kg")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = virtual_potential_temperature(t, q, p)

    for i, (t_vals, q_vals, p_val) in enumerate(zip(temperatures, specific_humidities, pressures)):
        ref = array.virtual_potential_temperature(np.array(t_vals), np.array(q_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "vptmp").all()


def test_fieldlist_temperature_from_potential_temperature():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import temperature_from_potential_temperature

    th_values = [[300.0, 290.0], [310.0, 305.0], [350.0, 340.0]]
    th = _make_fieldlist(th_values, variable="pt", units="K")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = temperature_from_potential_temperature(th, p)

    for i, (th_vals, p_val) in enumerate(zip(th_values, pressures)):
        ref = array.temperature_from_potential_temperature(np.array(th_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "t").all()


def test_fieldlist_temperature_from_potential_temperature_infer_pressure():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import temperature_from_potential_temperature

    th_values = [[300.0, 290.0], [310.0, 305.0], [350.0, 340.0]]
    th = _make_fieldlist_with_pressure(th_values, variable="pt", units="K")
    out = temperature_from_potential_temperature(th)

    for i, (th_vals, p_val) in enumerate(zip(th_values, pressures)):
        ref = array.temperature_from_potential_temperature(np.array(th_vals), p_val)
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "t").all()


def test_fieldlist_pressure_on_dry_adiabat():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import pressure_on_dry_adiabat

    t = _make_fieldlist(temperatures, variable="t", units="K")
    t_def = _make_fieldlist([[300.0, 295.0]] * 3, variable="t", units="K")
    p_def = _make_fieldlist([[100000.0, 100000.0]] * 3, variable="pres", units="Pa")
    out = pressure_on_dry_adiabat(t, t_def, p_def)

    for i, t_vals in enumerate(temperatures):
        ref = array.pressure_on_dry_adiabat(np.array(t_vals), np.array([300.0, 295.0]), np.array([100000.0, 100000.0]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "pres").all()
    assert (np.array(out.get("parameter.units")) == "Pa").all()


def test_fieldlist_temperature_on_dry_adiabat():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import temperature_on_dry_adiabat

    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    t_def = _make_fieldlist([[300.0]] * 3, variable="t", units="K")
    p_def = _make_fieldlist([[100000.0]] * 3, variable="pres", units="Pa")
    out = temperature_on_dry_adiabat(p, t_def, p_def)

    for i, p_val in enumerate(pressures):
        ref = array.temperature_on_dry_adiabat(np.array([p_val]), np.array([300.0]), np.array([100000.0]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "t").all()
    assert (np.array(out.get("parameter.units")) == "K").all()


def test_fieldlist_lcl_temperature():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import lcl_temperature

    t = _make_fieldlist(temperatures, variable="t", units="K")
    td = _make_fieldlist(dewpoints, variable="td", units="K")
    out = lcl_temperature(t, td)

    for i, (t_vals, td_vals) in enumerate(zip(temperatures, dewpoints)):
        ref = array.lcl_temperature(np.array(t_vals), np.array(td_vals))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "t_lcl").all()


def test_fieldlist_lcl():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import lcl

    t = _make_fieldlist(temperatures, variable="t", units="K")
    td = _make_fieldlist(dewpoints, variable="td", units="K")
    p = _make_fieldlist([[p_val, p_val] for p_val in pressures], variable="pres", units="Pa")
    t_out, p_out = lcl(t, td, p)

    for i, (t_vals, td_vals, p_val) in enumerate(zip(temperatures, dewpoints, pressures)):
        t_ref, p_ref = array.lcl(np.array(t_vals), np.array(td_vals), np.array([p_val, p_val]))
        np.testing.assert_allclose(t_out[i].values, t_ref)
        np.testing.assert_allclose(p_out[i].values, p_ref)
    assert (np.array(t_out.get("parameter.variable")) == "t_lcl").all()
    assert (np.array(p_out.get("parameter.variable")) == "p_lcl").all()
    assert (np.array(t_out.get("parameter.units")) == "K").all()
    assert (np.array(p_out.get("parameter.units")) == "Pa").all()


def test_fieldlist_ept_from_dewpoint():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import ept_from_dewpoint

    t = _make_fieldlist(temperatures, variable="t", units="K")
    td = _make_fieldlist(dewpoints, variable="td", units="K")
    p = _make_fieldlist([[p_val, p_val] for p_val in pressures], variable="pres", units="Pa")
    out = ept_from_dewpoint(t, td, p)

    for i, (t_vals, td_vals, p_val) in enumerate(zip(temperatures, dewpoints, pressures)):
        ref = array.ept_from_dewpoint(np.array(t_vals), np.array(td_vals), np.array([p_val, p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "eqpt").all()


def test_fieldlist_ept_from_specific_humidity():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import ept_from_specific_humidity

    t = _make_fieldlist(temperatures, variable="t", units="K")
    q = _make_fieldlist(specific_humidities, variable="q", units="kg/kg")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = ept_from_specific_humidity(t, q, p)

    for i, (t_vals, q_vals, p_val) in enumerate(zip(temperatures, specific_humidities, pressures)):
        ref = array.ept_from_specific_humidity(np.array(t_vals), np.array(q_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "eqpt").all()


def test_fieldlist_saturation_ept():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import saturation_ept

    t = _make_fieldlist(temperatures, variable="t", units="K")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = saturation_ept(t, p)

    for i, (t_vals, p_val) in enumerate(zip(temperatures, pressures)):
        ref = array.saturation_ept(np.array(t_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "sept").all()


def test_fieldlist_temperature_on_moist_adiabat():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import temperature_on_moist_adiabat

    ept_values = [[330.0, 320.0], [340.0, 335.0], [350.0, 345.0]]
    ept = _make_fieldlist(ept_values, variable="ept", units="K")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = temperature_on_moist_adiabat(ept, p)

    for i, (ept_vals, p_val) in enumerate(zip(ept_values, pressures)):
        ref = array.temperature_on_moist_adiabat(np.array(ept_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "t").all()


def test_fieldlist_wet_bulb_temperature_from_dewpoint():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import wet_bulb_temperature_from_dewpoint

    t = _make_fieldlist(temperatures, variable="t", units="K")
    td = _make_fieldlist(dewpoints, variable="td", units="K")
    p = _make_fieldlist([[p_val, p_val] for p_val in pressures], variable="pres", units="Pa")
    out = wet_bulb_temperature_from_dewpoint(t, td, p)

    for i, (t_vals, td_vals, p_val) in enumerate(zip(temperatures, dewpoints, pressures)):
        ref = array.wet_bulb_temperature_from_dewpoint(np.array(t_vals), np.array(td_vals), np.array([p_val, p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "wbgt").all()


def test_fieldlist_wet_bulb_temperature_from_specific_humidity():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import wet_bulb_temperature_from_specific_humidity

    t = _make_fieldlist(temperatures, variable="t", units="K")
    q = _make_fieldlist(specific_humidities, variable="q", units="kg/kg")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = wet_bulb_temperature_from_specific_humidity(t, q, p)

    for i, (t_vals, q_vals, p_val) in enumerate(zip(temperatures, specific_humidities, pressures)):
        ref = array.wet_bulb_temperature_from_specific_humidity(np.array(t_vals), np.array(q_vals), np.array([p_val]))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "wbgt").all()


def test_fieldlist_wet_bulb_potential_temperature_from_dewpoint():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import wet_bulb_potential_temperature_from_dewpoint

    t = _make_fieldlist(temperatures, variable="t", units="K")
    td = _make_fieldlist(dewpoints, variable="td", units="K")
    p = _make_fieldlist([[p_val, p_val] for p_val in pressures], variable="pres", units="Pa")
    out = wet_bulb_potential_temperature_from_dewpoint(t, td, p)

    for i, (t_vals, td_vals, p_val) in enumerate(zip(temperatures, dewpoints, pressures)):
        ref = array.wet_bulb_potential_temperature_from_dewpoint(
            np.array(t_vals), np.array(td_vals), np.array([p_val, p_val])
        )
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "wbgpt").all()


def test_fieldlist_wet_bulb_potential_temperature_from_specific_humidity():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import wet_bulb_potential_temperature_from_specific_humidity

    t = _make_fieldlist(temperatures, variable="t", units="K")
    q = _make_fieldlist(specific_humidities, variable="q", units="kg/kg")
    p = _make_fieldlist([[p_val] for p_val in pressures], variable="pres", units="Pa")
    out = wet_bulb_potential_temperature_from_specific_humidity(t, q, p)

    for i, (t_vals, q_vals, p_val) in enumerate(zip(temperatures, specific_humidities, pressures)):
        ref = array.wet_bulb_potential_temperature_from_specific_humidity(
            np.array(t_vals), np.array(q_vals), np.array([p_val])
        )
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "wbgpt").all()


def test_fieldlist_specific_gas_constant():
    import earthkit.meteo.thermo.array as array
    from earthkit.meteo.thermo.fieldlist import specific_gas_constant

    q = _make_fieldlist(specific_humidities, variable="q", units="kg/kg")
    out = specific_gas_constant(q)

    for i, q_vals in enumerate(specific_humidities):
        ref = array.specific_gas_constant(np.array(q_vals))
        np.testing.assert_allclose(out[i].values, ref)
    assert (np.array(out.get("parameter.variable")) == "R").all()
    assert (np.array(out.get("parameter.units")) == "J kg-1 K-1").all()
