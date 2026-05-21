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

U_WINDS = [[-5.0, 3.0], [10.0, -7.0], [2.0, 8.0]]  # m/s
V_WINDS = [[8.0, -4.0], [-6.0, 5.0], [-3.0, 1.0]]  # m/s
OMEGAS = [[-0.5, 0.3], [0.2, -0.1], [0.05, -0.3]]  # Pa/s
TEMPERATURES = [[293.15, 283.15], [273.15, 280.15], [285.54, 290.15]]  # K
PRESSURES = [100000.0, 85000.0, 50000.0]  # Pa

PARAMETERS = {
    "u": {"variable": "u", "units": "m/s"},
    "v": {"variable": "v", "units": "m/s"},
    "w": {"variable": "w", "units": "Pa/s"},
    "t": {"variable": "t", "units": "K"},
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

        return Field.from_components(
            values=np.array(values[0]),
            parameter={"variable": param_def["variable"], "units": param_def["units"]},
            vertical=vertical,
        )
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
def test_fieldlist_speed(input_type):
    import earthkit.meteo.wind.array as array_wind
    import earthkit.meteo.wind.fieldlist as wind

    u = _make_input_fieldlist("u", values=U_WINDS, input_type=input_type)
    v = _make_input_fieldlist("v", values=V_WINDS, input_type=input_type)
    out = wind.speed(u, v)

    assert isinstance(out, type(u))

    if input_type == "fieldlist":
        assert len(out) == len(u)
        assert out.get("parameter.variable") == ["ws"] * len(u)
        assert np.allclose(np.array(out.get("vertical.level")), np.array(PRESSURES) / 100.0)
        for f, u_vals, v_vals in zip(out, U_WINDS, V_WINDS):
            ref = array_wind.speed(np.array(u_vals), np.array(v_vals))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "ws"
        assert np.allclose(out.get("vertical.level"), PRESSURES[0] / 100.0)
        ref = array_wind.speed(np.array(U_WINDS[0]), np.array(V_WINDS[0]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_direction(input_type):
    import earthkit.meteo.wind.array as array_wind
    import earthkit.meteo.wind.fieldlist as wind

    u = _make_input_fieldlist("u", values=U_WINDS, input_type=input_type)
    v = _make_input_fieldlist("v", values=V_WINDS, input_type=input_type)
    out = wind.direction(u, v)

    assert isinstance(out, type(u))

    if input_type == "fieldlist":
        assert len(out) == len(u)
        assert out.get("parameter.variable") == ["wdir"] * len(u)
        assert np.allclose(np.array(out.get("vertical.level")), np.array(PRESSURES) / 100.0)
        for f, u_vals, v_vals in zip(out, U_WINDS, V_WINDS):
            ref = array_wind.direction(np.array(u_vals), np.array(v_vals))
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "wdir"
        assert np.allclose(out.get("vertical.level"), PRESSURES[0] / 100.0)
        ref = array_wind.direction(np.array(U_WINDS[0]), np.array(V_WINDS[0]))
        np.testing.assert_allclose(out.values, ref)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_xy_to_polar(input_type):
    import earthkit.meteo.wind.array as array_wind
    import earthkit.meteo.wind.fieldlist as wind

    u = _make_input_fieldlist("u", values=U_WINDS, input_type=input_type)
    v = _make_input_fieldlist("v", values=V_WINDS, input_type=input_type)
    out_speed, out_dir = wind.xy_to_polar(u, v)

    assert isinstance(out_speed, type(u))
    assert isinstance(out_dir, type(u))

    if input_type == "fieldlist":
        assert len(out_speed) == len(u)
        assert len(out_dir) == len(u)
        assert out_speed.get("parameter.variable") == ["ws"] * len(u)
        assert out_dir.get("parameter.variable") == ["wdir"] * len(u)
        for f_s, f_d, u_vals, v_vals in zip(out_speed, out_dir, U_WINDS, V_WINDS):
            ref_s = array_wind.speed(np.array(u_vals), np.array(v_vals))
            ref_d = array_wind.direction(np.array(u_vals), np.array(v_vals))
            np.testing.assert_allclose(f_s.values, ref_s)
            np.testing.assert_allclose(f_d.values, ref_d)
    elif input_type == "field":
        assert out_speed.get("parameter.variable") == "ws"
        assert out_dir.get("parameter.variable") == "wdir"
        ref_s = array_wind.speed(np.array(U_WINDS[0]), np.array(V_WINDS[0]))
        ref_d = array_wind.direction(np.array(U_WINDS[0]), np.array(V_WINDS[0]))
        np.testing.assert_allclose(out_speed.values, ref_s)
        np.testing.assert_allclose(out_dir.values, ref_d)


def test_fieldlist_polar_to_xy():
    import earthkit.meteo.wind.fieldlist as wind

    u = _make_input_fieldlist("u", values=U_WINDS, input_type="fieldlist")
    v = _make_input_fieldlist("v", values=V_WINDS, input_type="fieldlist")

    sp, dr = wind.xy_to_polar(u, v)
    out_u, out_v = wind.polar_to_xy(sp, dr)

    assert len(out_u) == len(u)
    assert len(out_v) == len(v)

    for f_u, f_v, u_vals, v_vals in zip(out_u, out_v, U_WINDS, V_WINDS):
        np.testing.assert_allclose(f_u.values, np.array(u_vals), atol=1e-10)
        np.testing.assert_allclose(f_v.values, np.array(v_vals), atol=1e-10)


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
@pytest.mark.parametrize("pres_type", ["fl", "value", None])
def test_fieldlist_w_from_omega(input_type, pres_type):
    import earthkit.meteo.wind.array as array_wind
    import earthkit.meteo.wind.fieldlist as wind

    omega = _make_input_fieldlist("w", values=OMEGAS, input_type=input_type)
    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type)
    p = _make_pres_fieldlist(omega, pres_type=pres_type)
    out = wind.w_from_omega(omega, t, p=p)

    assert isinstance(out, type(omega))

    if input_type == "fieldlist":
        assert len(out) == len(omega)
        assert out.get("parameter.variable") == ["wz"] * len(omega)
        assert out.get("parameter.units") == ["m/s"] * len(omega)
        for f, omega_vals, t_vals, p_val in zip(out, OMEGAS, TEMPERATURES, PRESSURES):
            ref = array_wind.w_from_omega(np.array(omega_vals), np.array(t_vals), p_val)
            np.testing.assert_allclose(f.values, ref)
    elif input_type == "field":
        assert out.get("parameter.variable") == "wz"
        assert out.get("parameter.units") == "m/s"
        ref = array_wind.w_from_omega(np.array(OMEGAS[0]), np.array(TEMPERATURES[0]), PRESSURES[0])
        np.testing.assert_allclose(out.values, ref)
