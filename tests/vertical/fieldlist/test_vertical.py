# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from importlib import import_module

import numpy as np
import pytest

# import earthkit.meteo.vertical.array as vertical
from earthkit.meteo.utils.testing import NO_EKD, Tolerance

np.set_printoptions(formatter={"float_kind": "{:.15f}".format})
pytestmark = pytest.mark.skipif(NO_EKD, reason="EKD is not installed")


def _get_data(name):
    # import os
    # import sys

    # # here = os.path.dirname(__file__)
    # # sys.path.insert(0, here)
    # sys.path.insert(0, "../")

    return import_module(name)


DATA_HYBRID_CORE = _get_data("_hybrid_core_data")
DATA_HYBRID_H = _get_data("_hybrid_height_data")


# np.set_printoptions(formatter={"float_kind": "{:.10f}".format})
# pytestmark = pytest.mark.skipif(NO_EKD, reason="EKD is not installed")

# U_WINDS = [[-5.0, 3.0], [10.0, -7.0], [2.0, 8.0]]  # m/s
# V_WINDS = [[8.0, -4.0], [-6.0, 5.0], [-3.0, 1.0]]  # m/s
# OMEGAS = [[-0.5, 0.3], [0.2, -0.1], [0.05, -0.3]]  # Pa/s
# TEMPERATURES = [[293.15, 283.15], [273.15, 280.15], [285.54, 290.15]]  # K
PRESSURES = [100000.0, 85000.0]  # Pa

PARAMETERS = {
    "z": {"variable": "z", "units": "m2/s2"},  # geopotential
    "gh": {"variable": "gh", "units": "gpm"},  # geopotential height
    "h": {"variable": "h", "units": "m"},  # geometric height
    "sp": {"variable": "sp", "units": "Pa"},  # surface pressure
}


def _make_input_fieldlist(param, values, input_type="fieldlist", level_type="pressure"):
    from earthkit.data import Field, FieldList

    param_def = PARAMETERS[param]

    if input_type == "field":
        if level_type == "pressure":
            vertical = {"level": PRESSURES[0] / 100, "level_type": "pressure"}
        elif level_type == "surface":
            vertical = {"level": 0, "level_type": "surface"}
        elif level_type == "hybrid1":
            vertical = {"level": 1, "level_type": "hybrid"}
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

        elif level_type == "hybrid1":
            fl = []
            for v in values:
                vertical = {"level": 1, "level_type": "hybrid"}
                fl.append(
                    Field.from_components(
                        values=np.array(v),
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
def test_fieldlist_geopotential_height_from_geopotential(input_type):
    import earthkit.meteo.vertical.fieldlist as vertical

    z_values = np.asarray([[1000.0, 10000.0], [20000.0, 15000.0]])  # m2/s2
    ref_values = np.asarray([[101.97162129779284, 1019.7162129779283], [2039.4324259558566, 1529.5743194668923]])

    z = _make_input_fieldlist("z", z_values, input_type=input_type, level_type="pressure")
    out = vertical.geopotential_height_from_geopotential(z)

    assert isinstance(out, type(z))

    if input_type == "fieldlist":
        assert len(out) == len(z)
        assert out.get("parameter.variable") == ["gh"] * len(z)
        assert np.allclose(np.array(out.get("vertical.level")), np.array(PRESSURES) / 100.0)
        assert np.allclose(out.to_numpy(), ref_values)
    elif input_type == "field":
        assert out.get("parameter.variable") == "gh"
        assert np.allclose(out.get("vertical.level"), PRESSURES[0] / 100.0)
        np.testing.assert_allclose(out.to_numpy(), ref_values[0])


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_geopotential_from_geopotential_height(input_type):
    import earthkit.meteo.vertical.fieldlist as vertical

    gh_values = np.asarray([[100.0, 1000.0], [2000.0, 1500.0]])  # gpm
    ref_values = np.asarray([[980.665, 9806.65], [19613.3, 14709.974999999999]])

    gh = _make_input_fieldlist("gh", gh_values, input_type=input_type, level_type="pressure")
    out = vertical.geopotential_from_geopotential_height(gh)

    assert isinstance(out, type(gh))

    if input_type == "fieldlist":
        assert len(out) == len(gh)
        assert out.get("parameter.variable") == ["z"] * len(gh)
        assert np.allclose(np.array(out.get("vertical.level")), np.array(PRESSURES) / 100.0)
        assert np.allclose(out.to_numpy(), ref_values)
    elif input_type == "field":
        assert out.get("parameter.variable") == "z"
        assert np.allclose(out.get("vertical.level"), PRESSURES[0] / 100.0)
        np.testing.assert_allclose(out.to_numpy(), ref_values[0])


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_geopotential_from_geometric_height(input_type):
    import earthkit.meteo.vertical.fieldlist as vertical

    h_values = np.asarray([[100.0, 1000.0], [2000.0, 1500.0]])  # m
    ref_values = np.asarray([[980.6496081563203, 9805.111033023139], [19607.14509798722, 14706.51259598125]])

    h = _make_input_fieldlist("h", h_values, input_type=input_type, level_type="pressure")
    out = vertical.geopotential_from_geometric_height(h)

    assert isinstance(out, type(h))

    if input_type == "fieldlist":
        assert len(out) == len(h)
        assert out.get("parameter.variable") == ["z"] * len(h)
        assert np.allclose(np.array(out.get("vertical.level")), np.array(PRESSURES) / 100.0)
        assert np.allclose(out.to_numpy(), ref_values)
    elif input_type == "field":
        assert out.get("parameter.variable") == "z"
        assert np.allclose(out.get("vertical.level"), PRESSURES[0] / 100.0)
        np.testing.assert_allclose(out.to_numpy(), ref_values[0])


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_geopotential_height_from_geometric_height(input_type):
    import earthkit.meteo.vertical.fieldlist as vertical

    h_values = np.asarray([[100.0, 1000.0], [2000.0, 1500.0]])  # m
    ref_values = np.asarray([[99.99843046874521, 999.8430690422457], [1999.372374662828, 1499.6469330486202]])

    h = _make_input_fieldlist("h", h_values, input_type=input_type, level_type="pressure")
    out = vertical.geopotential_height_from_geometric_height(h)

    assert isinstance(out, type(h))

    if input_type == "fieldlist":
        assert len(out) == len(h)
        assert out.get("parameter.variable") == ["gh"] * len(h)
        assert np.allclose(np.array(out.get("vertical.level")), np.array(PRESSURES) / 100.0)
        assert np.allclose(out.to_numpy(), ref_values)
    elif input_type == "field":
        assert out.get("parameter.variable") == "gh"
        assert np.allclose(out.get("vertical.level"), PRESSURES[0] / 100.0)
        np.testing.assert_allclose(out.to_numpy(), ref_values[0])


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_geometric_height_from_geopotential(input_type):
    import earthkit.meteo.vertical.fieldlist as vertical

    z_values = np.asarray([[1000.0, 10000.0], [20000.0, 15000.0]])  # m2/s2
    ref_values = np.asarray([[101.9732533813322, 1019.8794448449968], [2040.0854579587374, 1529.9416205658015]])

    z = _make_input_fieldlist("z", z_values, input_type=input_type, level_type="pressure")
    out = vertical.geometric_height_from_geopotential(z)

    assert isinstance(out, type(z))

    if input_type == "fieldlist":
        assert len(out) == len(z)
        assert out.get("parameter.variable") == ["h"] * len(z)
        assert np.allclose(np.array(out.get("vertical.level")), np.array(PRESSURES) / 100.0)
        assert np.allclose(out.to_numpy(), ref_values)
    elif input_type == "field":
        assert out.get("parameter.variable") == "h"
        assert np.allclose(out.get("vertical.level"), PRESSURES[0] / 100.0)
        np.testing.assert_allclose(out.to_numpy(), ref_values[0])


@pytest.mark.parametrize("input_type", ["fieldlist", "field"])
def test_fieldlist_geometric_height_from_geopotential_height(input_type):
    import earthkit.meteo.vertical.fieldlist as vertical

    gh_values = np.asarray([[100.0, 1000.0], [2000.0, 1500.0]])  # gpm
    ref_values = np.asarray([[100.0015695805249, 1000.1569802278693], [2000.6280194981214, 1500.3532332380232]])

    gh = _make_input_fieldlist("gh", gh_values, input_type=input_type, level_type="pressure")
    out = vertical.geometric_height_from_geopotential_height(gh)

    assert isinstance(out, type(gh))

    if input_type == "fieldlist":
        assert len(out) == len(gh)
        assert out.get("parameter.variable") == ["h"] * len(gh)
        assert np.allclose(np.array(out.get("vertical.level")), np.array(PRESSURES) / 100.0)
        assert np.allclose(out.to_numpy(), ref_values)
    elif input_type == "field":
        assert out.get("parameter.variable") == "h"
        assert np.allclose(out.get("vertical.level"), PRESSURES[0] / 100.0)
        np.testing.assert_allclose(out.to_numpy(), ref_values[0])


def test_fieldlist_pressure_on_hybrid_levels_core():
    import earthkit.meteo.vertical.fieldlist as vertical

    sp = [DATA_HYBRID_CORE.p_surf]
    A = DATA_HYBRID_CORE.A
    B = DATA_HYBRID_CORE.B
    ref_p_full = DATA_HYBRID_CORE.p_full
    ref_p_half = DATA_HYBRID_CORE.p_half
    ref_delta = DATA_HYBRID_CORE.delta
    ref_alpha = DATA_HYBRID_CORE.alpha

    sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B = (
        np.asarray(x) for x in [sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B]
    )

    sp = _make_input_fieldlist("sp", sp, input_type="field", level_type="hybrid1")

    p_full, p_half, delta, alpha = vertical.pressure_on_hybrid_levels(
        sp, A=A, B=B, alpha_top="ifs", output=["full", "half", "delta", "alpha"]
    )

    # print("p_full", repr(p_full))
    # print("p_half", repr(p_half))
    # print("delta", repr(delta))
    # print("alpha", repr(alpha))

    # print("p_full diff", repr(xp.max(xp.abs(p_full - ref_p_full))))
    # print("p_half diff", repr(xp.max(xp.abs(p_half - ref_p_half))))
    # print("delta diff", repr(xp.max(xp.abs(delta - ref_delta))))
    # print("alpha diff", repr(xp.max(xp.abs(alpha - ref_alpha))))

    tolerance = Tolerance({
        "p_full": {64: (1e-8, 1e-6)},
        "p_half": {64: (1e-8, 1e-6)},
        "delta": {64: (1e-8, 1e-6), 32: (1e-6, 1e-5)},
        "alpha": {64: (1e-8, 1e-6), 32: (1e-4, 1e-5)},
    })

    print("p_full", len(p_full), type(p_full), p_full.to_numpy().shape)

    atol, rtol = tolerance.get(key="p_full")
    np.testing.assert_allclose(p_full.to_numpy(), ref_p_full, atol=atol, rtol=rtol)

    atol, rtol = tolerance.get(key="p_half")
    np.testing.assert_allclose(p_half.to_numpy(), ref_p_half, atol=atol, rtol=rtol)

    # for i in range(delta.shape[0]):
    #     print(
    #         f"delta level {i}: computed={delta[i]}, "
    #         f"reference={ref_delta[i]} diff={delta[i]-ref_delta[i]}"
    #     )

    atol, rtol = tolerance.get(key="delta")
    assert np.allclose(delta.to_numpy(), ref_delta, atol=atol, rtol=rtol)

    atol, rtol = tolerance.get(key="alpha")
    assert np.allclose(alpha.to_numpy(), ref_alpha, atol=atol, rtol=rtol)
