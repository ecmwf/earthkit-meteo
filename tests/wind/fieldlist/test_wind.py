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
LATITUDES = [45.0, 90.0]  # degrees
LONGITUDES = [90.0, 180.0]  # degrees

REF_SPEED = [
    [9.433981132056603, 5.0],
    [11.661903789690601, 8.602325267042627],
    [3.605551275463989, 8.06225774829855],
]

REF_DIRECTION = [
    [147.9946167919165, 323.130102354156],
    [300.9637565320735, 125.53767779197437],
    [326.3099324740202, 262.8749836510982],
]

REF_W_FROM_OMEGA = [
    [0.042905350479011686, -0.024865051996859275],
    [-0.018813249965734535, 0.00964768804301763],
    [-0.008358310609433397, 0.05095952560048538],
]

REF_CORIOLIS = [0.00010312608048829148, 0.00014584230166092124]


PARAMETERS = {
    "u": {"variable": "u", "units": "m/s"},
    "v": {"variable": "v", "units": "m/s"},
    "w": {"variable": "w", "units": "Pa/s"},
    "t": {"variable": "t", "units": "K"},
}


def _make_input_fieldlist(param, values, input_type="fieldlist", level_type="pressure", geo=False):
    from earthkit.data import Field, FieldList

    param_def = PARAMETERS[param]

    geo_kwarg = {}
    if geo:
        geo_kwarg = {"geography": {"latitudes": LATITUDES, "longitudes": LONGITUDES}}

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
            **geo_kwarg,
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
                        **geo_kwarg,
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
                    **geo_kwarg,
                )
            )
            return FieldList.from_fields(fl)


def _make_pres_fieldlist(md_fl, pres_type="fl", geo=False):
    from earthkit.data import Field, FieldList

    geo_kwarg = {}
    if geo:
        geo_kwarg = {"geography.latitudes": LATITUDES, "geography.longitudes": LONGITUDES}

    if isinstance(md_fl, FieldList):
        if pres_type == "fl":
            p_fl = []
            for md_field in md_fl:
                p_fl.append(
                    md_field.set(
                        {"parameter.variable": "pres"},
                        geo_kwarg,
                        values=md_field.values * 0.0 + md_field.vertical.level(units="Pa"),
                    ).sync()
                )
            return FieldList.from_fields(p_fl)
        elif pres_type == "value":
            pres = np.array(md_fl.get("vertical.level")) * 100.0  # convert to Pa
            return pres
    elif isinstance(md_fl, Field):
        if pres_type == "fl":
            return md_fl.set(
                {"parameter.variable": "pres"},
                geo_kwarg,
                values=md_fl.values * 0.0 + md_fl.vertical.level(units="Pa"),
            ).sync()
        elif pres_type == "value":
            pres = md_fl.vertical.level(units="Pa")
            return pres
    else:
        raise ValueError(f"Unsupported md_fl type: {type(md_fl)}")

    if pres_type is None:
        return None

    raise ValueError(f"Unsupported pres_type: {pres_type}")


def _add_geo_to_fieldlist(fl):
    from earthkit.data import Field, FieldList

    if isinstance(fl, FieldList):
        r = []
        for field in fl:
            r.append(field.set({"geography.latitudes": LATITUDES, "geography.longitudes": LONGITUDES}))
        return FieldList.from_fields(r)
    elif isinstance(fl, Field):
        return fl.set({"geography.latitudes": LATITUDES, "geography.longitudes": LONGITUDES})
    else:
        raise ValueError(f"Unsupported fl type: {type(fl)}")


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_speed(input_type):
    import earthkit.meteo.wind.fieldlist as wind

    u = _make_input_fieldlist("u", values=U_WINDS, input_type=input_type)
    v = _make_input_fieldlist("v", values=V_WINDS, input_type=input_type)
    out = wind.speed(u, v)

    assert isinstance(out, type(u))

    if input_type == "fieldlist":
        assert len(out) == len(u)
        assert out.get("parameter.variable") == ["ws"] * len(u)
        assert np.allclose(np.array(out.get("vertical.level")), np.array(PRESSURES) / 100.0)
        for f, ref_vals in zip(out, REF_SPEED):
            np.testing.assert_allclose(f.values, np.array(ref_vals))
    elif input_type == "field":
        assert out.get("parameter.variable") == "ws"
        assert np.allclose(out.vertical.level(units="Pa"), PRESSURES[0])
        np.testing.assert_allclose(out.values, np.array(REF_SPEED[0]))


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_direction(input_type):
    import earthkit.meteo.wind.fieldlist as wind

    u = _make_input_fieldlist("u", values=U_WINDS, input_type=input_type)
    v = _make_input_fieldlist("v", values=V_WINDS, input_type=input_type)
    out = wind.direction(u, v)

    assert isinstance(out, type(u))

    if input_type == "fieldlist":
        assert len(out) == len(u)
        assert out.get("parameter.variable") == ["wdir"] * len(u)
        assert np.allclose(np.array(out.get("vertical.level")), np.array(PRESSURES) / 100.0)
        for f, ref_vals in zip(out, REF_DIRECTION):
            np.testing.assert_allclose(f.values, np.array(ref_vals))
    elif input_type == "field":
        assert out.get("parameter.variable") == "wdir"
        assert np.allclose(out.vertical.level(units="Pa"), PRESSURES[0])
        np.testing.assert_allclose(out.values, np.array(REF_DIRECTION[0]))


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_xy_to_polar(input_type):
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
        for f_s, f_d, ref_s, ref_d in zip(out_speed, out_dir, REF_SPEED, REF_DIRECTION):
            np.testing.assert_allclose(f_s.values, np.array(ref_s))
            np.testing.assert_allclose(f_d.values, np.array(ref_d))
    elif input_type == "field":
        assert out_speed.get("parameter.variable") == "ws"
        assert out_dir.get("parameter.variable") == "wdir"
        np.testing.assert_allclose(out_speed.values, np.array(REF_SPEED[0]))
        np.testing.assert_allclose(out_dir.values, np.array(REF_DIRECTION[0]))


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
        for f, ref_vals in zip(out, REF_W_FROM_OMEGA):
            np.testing.assert_allclose(f.values, np.array(ref_vals))
    elif input_type == "field":
        assert out.get("parameter.variable") == "wz"
        assert out.get("parameter.units") == "m/s"
        np.testing.assert_allclose(out.values, np.array(REF_W_FROM_OMEGA[0]))


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_coriolis(input_type):
    import earthkit.meteo.wind.fieldlist as wind

    t = _make_input_fieldlist("t", values=TEMPERATURES, input_type=input_type, geo=True)
    out = wind.coriolis(t)

    assert isinstance(out, type(t))

    if input_type == "fieldlist":
        assert len(out) == len(t)
        assert out.get("parameter.variable") == ["fc"] * len(t)
        assert out.get("parameter.units") == ["1/s"] * len(t)
        for f in out:
            np.testing.assert_allclose(f.values, np.array(REF_CORIOLIS))
    elif input_type == "field":
        assert out.get("parameter.variable") == "fc"
        assert out.get("parameter.units") == "1/s"
        np.testing.assert_allclose(out.values, np.array(REF_CORIOLIS))
