# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import numpy as np
import pytest

from earthkit.meteo.utils import testing
from earthkit.meteo.utils.testing import NO_EKD

# import earthkit.meteo.vertical.array as vertical

np.set_printoptions(formatter={"float_kind": "{:.15f}".format})
pytestmark = pytest.mark.skipif(NO_EKD, reason="EKD is not installed")


def _get_fieldlist(name, sample=False):
    import earthkit.data as ekd

    if sample:
        return ekd.from_source("sample", name).to_fieldlist()
    else:
        path = testing.get_test_data(name, "test-data")
        fl = ekd.from_source("file", path).to_fieldlist()
        return fl


def test_fieldlist_grib_pressure_on_hybrid_levels_core():
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tq_ml137.grib2")

    sp = ds.sel({"parameter.variable": "lnsp", "vertical.level": 1})[0]
    sp = sp.set(values=np.exp(sp.values))

    p_full, p_half, delta, alpha = vertical.pressure_on_hybrid_levels(
        sp, alpha_top="ifs", output=["full", "half", "delta", "alpha"]
    )

    data = {
        "p_full": (
            p_full,
            np.arange(1, 138),
            (1e-8, 1e-6),
            {"parameter.variable": "pres", "vertical.level": 1, "vertical.level_type": "hybrid"},
            np.array([
                [101170.51199572402, 100410.7716802512],
                [101238.45934072728, 87011.05993156167],
            ]),
        ),
        "p_half": (
            p_half,
            np.arange(0, 138),
            (1e-8, 1e-6),
            {"parameter.variable": "pres_half", "vertical.level": 0, "vertical.level_type": "hybrid"},
            np.array([
                [101290.53523679737, 100529.89360637378],
                [101358.56319086751, 87114.28516207798],
            ]),
        ),
        "delta": (
            delta,
            np.arange(1, 138),
            (1e-8, 1e-6),
            {"parameter.variable": "hybrid_delta", "vertical.level": 1, "vertical.level_type": "hybrid"},
            np.array([
                [0.0023726932880611053, 0.0023726932880608837],
                [0.0023726932880611053, 0.0023726932880608837],
            ]),
        ),
        "alpha": (
            alpha,
            np.arange(1, 138),
            (1e-8, 1e-6),
            {"parameter.variable": "hybrid_alpha", "vertical.level": 1, "vertical.level_type": "hybrid"},
            np.array([
                [0.0011858775045937575, 0.0011858775046401648],
                [0.001185877504600863, 0.0011858775046598158],
            ]),
        ),
    }

    for key, (fl, ref_levels, (atol, rtol), ref_metadata, ref_val) in data.items():
        assert isinstance(fl, FieldList)
        assert len(fl) == len(ref_levels), f"Expected {len(ref_levels)} fields in {key}, but got {len(fl)}."
        assert (
            np.testing.assert_allclose(np.array(fl.get("vertical.level")), ref_levels, atol=atol, rtol=rtol) is None
        ), f"Vertical levels mismatch in {key}"

        assert fl.get("vertical.level_type") == ["hybrid"] * len(ref_levels)
        field = fl[0]

        for k, v in ref_metadata.items():
            assert field.get(k) == v, f"Metadata mismatch for {key}: expected {k}={v}, but got {field.get(k)}"

        np.testing.assert_allclose(fl[-1].to_numpy()[:2, :2], ref_val, atol=atol, rtol=rtol)


@pytest.mark.parametrize("sort_mode", [(None, None), ("ascending", "descending"), ("descending", "ascending")])
def test_fieldlist_grib_relative_geopotential_thickness_on_hybrid_levels(sort_mode):
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tq_ml137.grib2")

    sp = ds.sel({"parameter.variable": "lnsp", "vertical.level": 1})[0]
    sp = sp.set(values=np.exp(sp.values))

    t = ds.sel({"parameter.variable": "t"})
    if sort_mode[0] is not None:
        t = t.order_by({"vertical.level": sort_mode[0]})

    q = ds.sel({"parameter.variable": "q"})
    if sort_mode[1] is not None:
        q = q.order_by({"vertical.level": sort_mode[1]})

    out = vertical.relative_geopotential_thickness_on_hybrid_levels(t, q, sp)

    assert isinstance(out, FieldList)
    assert len(out) == len(t)
    assert out.get("vertical.level_type") == ["hybrid"] * len(t)
    assert out.get("vertical.level") == list(range(1, len(t) + 1))

    # top of the atmosphere is at level 1, so we check the first field in the output
    field = out[0]
    np.testing.assert_allclose(
        field.to_numpy()[:2, :2],
        [[791285.5729044699, 790887.5659324717], [773062.2997916321, 760702.9407670180]],
        atol=1e-8,
        rtol=1e-6,
    )

    # lowest model level just above the surface is at level 137, so we check the last field in the output
    field = out[-1]
    np.testing.assert_allclose(
        field.to_numpy()[:2, :2],
        [[97.2627107953, 98.3924258259], [102.5735530759, 101.7227610877]],
        atol=1e-8,
        rtol=1e-6,
    )


@pytest.mark.parametrize("sort_mode", [(None, None), ("ascending", "descending"), ("descending", "ascending")])
def test_fieldlist_grib_relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta(sort_mode):
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tq_ml137.grib2")

    sp = ds.sel({"parameter.variable": "lnsp", "vertical.level": 1})[0]
    sp = sp.set(values=np.exp(sp.values))

    alpha, delta = vertical.pressure_on_hybrid_levels(sp, output=["alpha", "delta"])

    t = ds.sel({"parameter.variable": "t"})
    if sort_mode[0] is not None:
        t = t.order_by({"vertical.level": sort_mode[0]})
    q = ds.sel({"parameter.variable": "q"})

    if sort_mode[1] is not None:
        q = q.order_by({"vertical.level": sort_mode[1]})

    out = vertical.relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta(t, q, alpha, delta)

    assert isinstance(out, FieldList)
    assert len(out) == len(t)
    assert out.get("vertical.level_type") == ["hybrid"] * len(t)
    assert out.get("vertical.level") == list(range(1, len(t) + 1))

    # top of the atmosphere is at level 1, so we check the first field in the output
    field = out[0]
    np.testing.assert_allclose(
        field.to_numpy()[:2, :2],
        [[791285.5729044699, 790887.5659324717], [773062.2997916321, 760702.9407670180]],
        atol=1e-8,
        rtol=1e-6,
    )

    # lowest model level just above the surface is at level 137, so we check the last field in the output
    field = out[-1]
    np.testing.assert_allclose(
        field.to_numpy()[:2, :2],
        [[97.2627107953, 98.3924258259], [102.5735530759, 101.7227610877]],
        atol=1e-8,
        rtol=1e-6,
    )


@pytest.mark.parametrize("sort_mode", [(None, None), ("ascending", "descending"), ("descending", "ascending")])
def test_fieldlist_grib_geopotential_on_hybrid_levels(sort_mode):
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tq_ml137.grib2")

    sp = ds.sel({"parameter.variable": "lnsp", "vertical.level": 1})[0]
    sp = sp.set(values=np.exp(sp.values))

    t = ds.sel({"parameter.variable": "t"})
    if sort_mode[0] is not None:
        t = t.order_by({"vertical.level": sort_mode[0]})

    q = ds.sel({"parameter.variable": "q"})
    if sort_mode[1] is not None:
        q = q.order_by({"vertical.level": sort_mode[1]})

    zs = ds.sel({"parameter.variable": "z", "vertical.level": 1})[0]

    out = vertical.geopotential_on_hybrid_levels(t, q, zs, sp)

    assert isinstance(out, FieldList)
    assert len(out) == len(t)
    assert out.get("vertical.level_type") == ["hybrid"] * len(t)
    assert out.get("vertical.level") == list(range(1, len(t) + 1))

    # top of the atmosphere is at level 1, so we check the first field in the output
    field = out[0]
    np.testing.assert_allclose(
        field.to_numpy()[:2, :2],
        [[790480.6981486105, 791162.6911766123], [773053.4250357727, 773295.0660111586]],
        atol=1e-8,
        rtol=1e-6,
    )

    # lowest model level just above the surface is at level 137, so we check the last field in the output
    field = out[-1]
    np.testing.assert_allclose(
        field.to_numpy()[:2, :2],
        [[-707.6120450640693, 373.51766996650116], [93.69879721648394, 12693.848005228374]],
        atol=1e-8,
        rtol=1e-6,
    )


@pytest.mark.parametrize("sort_mode", [(None, None), ("ascending", "descending"), ("descending", "ascending")])
def test_fieldlist_grib_height_on_hybrid_levels(sort_mode):
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tq_ml137.grib2")

    sp = ds.sel({"parameter.variable": "lnsp", "vertical.level": 1})[0]
    sp = sp.set(values=np.exp(sp.values))

    t = ds.sel({"parameter.variable": "t"})
    if sort_mode[0] is not None:
        t = t.order_by({"vertical.level": sort_mode[0]})

    q = ds.sel({"parameter.variable": "q"})
    if sort_mode[1] is not None:
        q = q.order_by({"vertical.level": sort_mode[1]})

    zs = ds.sel({"parameter.variable": "z", "vertical.level": 1})[0]

    out = vertical.height_on_hybrid_levels(t, q, zs, sp)

    assert isinstance(out, FieldList)
    assert len(out) == len(t)
    assert out.get("vertical.level_type") == ["hybrid"] * len(t)
    assert out.get("vertical.level") == list(range(1, len(t) + 1))
    assert out.get("parameter.variable") == ["h"] * len(t)
    assert out.get("parameter.units") == ["m"] * len(t)

    # top of the atmosphere is at level 1, so we check the first field in the output
    field = out[0]
    np.testing.assert_allclose(
        field.to_numpy()[:2, :2],
        [[81721.5462590315, 81682.7562239972], [79817.9712713831, 78558.0299056925]],
        atol=1e-8,
        rtol=1e-6,
    )

    # lowest model level just above the surface is at level 137, so we check the last field in the output
    field = out[-1]
    np.testing.assert_allclose(
        field.to_numpy()[:2, :2],
        [[9.9177962264, 10.0333393461], [10.4596057095, 10.3770340567]],
        atol=1e-8,
        rtol=1e-6,
    )


@pytest.mark.parametrize("sort_mode", [None, "ascending", "descending"])
@pytest.mark.parametrize(
    "target_p, ref_values",
    [
        (
            [50000.0, 1000000.0, 85000.0],
            np.array([
                [[256.84623176888516, 253.75744555161938], [268.90648252133036, 268.83196698107815]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[279.2465996835962, 277.5093980166771], [291.93205593367657, 294.0204548739655]],
            ]),
        ),
    ],
)
def test_fieldlist_grib_interpolate_hybrid_to_pressure_levels_core(sort_mode, target_p, ref_values):
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tq_ml137.grib2")

    sp = ds.sel({"parameter.variable": "lnsp", "vertical.level": 1})[0]
    sp = sp.set(values=np.exp(sp.values))

    t = ds.sel({"parameter.variable": "t"})
    if sort_mode is not None:
        t = t.order_by({"vertical.level": sort_mode})

    out = vertical.interpolate_hybrid_to_pressure_levels(t, target_p, sp)

    assert isinstance(out, FieldList)
    assert len(out) == len(target_p)
    assert out.get("vertical.level_type") == ["pressure"] * len(out)
    assert np.allclose(np.asarray(out.get("vertical.level")) * 100.0, target_p)
    assert out.get("parameter.variable") == ["t"] * len(target_p)
    assert out.get("parameter.units") == ["K"] * len(target_p)

    actual = out.to_numpy()[:, :2, :2]
    np.testing.assert_allclose(actual, ref_values, atol=1e-8, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("sort_mode", [None, "ascending", "descending"])
@pytest.mark.parametrize("aux_type", ["field", "fieldlist", "value"])
@pytest.mark.parametrize(
    "levels,target_p, aux,ref_values",
    [
        (
            None,
            [50000.0, 104000.0, 85000.0, 8000.0, 120000.0, -1000.0],
            {"aux_bottom_data": 300, "aux_bottom_p": 110000, "aux_top_data": 50, "aux_top_p": 5000},
            np.array([
                [[256.84623176888516, 253.75744555161938], [268.90648252133036, 268.83196698107815]],
                [[289.4319621110, 292.3333356379], [298.9159596310, 299.0067484103]],
                [[279.2465996835962, 277.5093980166771], [291.93205593367657, 294.0204548739655]],
                [[222.0212359444, 222.4824067040], [198.4045735869, 198.2838580423]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ]),
        ),
        (
            list(range(90, 138)),
            [50000.0, 104000.0, 85000.0, 25000.0, 120000.0, 10000.0],
            {"aux_bottom_data": 300, "aux_bottom_p": 110000, "aux_top_data": 50, "aux_top_p": 20000},
            np.array([
                [[256.84623176888516, 253.75744555161938], [268.90648252133036, 268.83196698107815]],
                [[289.4319621110, 292.3333356379], [298.9159596310, 299.0067484103]],
                [[279.2465996835962, 277.5093980166771], [291.93205593367657, 294.0204548739655]],
                [[99.0439875602, 98.2606462834], [102.1651469625, 109.6831539810]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ]),
        ),
    ],
)
def test_fieldlist_grib_interpolate_hybrid_to_pressure_levels_aux(
    sort_mode, aux_type, levels, target_p, aux, ref_values
):
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tq_ml137.grib2")

    sp = ds.sel({"parameter.variable": "lnsp", "vertical.level": 1})[0]
    sp = sp.set(values=np.exp(sp.values))

    t = ds.sel({"parameter.variable": "t"})
    if sort_mode is not None:
        t = t.order_by({"vertical.level": sort_mode})

    if levels is not None:
        r = []
        for f in t:
            if f.get("vertical.level") in levels:
                r.append(f)
        t = FieldList.from_fields(r)

    aux = dict(aux)
    if aux_type == "fieldlist":
        for key, value in aux.items():
            aux[key] = FieldList.from_fields([sp.set(values=np.full_like(sp.values, value))])
    elif aux_type == "field":
        for key, value in aux.items():
            aux[key] = sp.set(values=np.full_like(sp.values, value))
    elif aux_type != "value":
        raise ValueError(f"Unsupported aux_type: {aux_type}")

    out = vertical.interpolate_hybrid_to_pressure_levels(t, target_p, sp, **aux)

    assert isinstance(out, FieldList)
    assert len(out) == len(target_p)
    assert out.get("vertical.level_type") == ["pressure"] * len(out)
    assert np.allclose(np.asarray(out.get("vertical.level")) * 100.0, target_p)
    assert out.get("parameter.variable") == ["t"] * len(target_p)
    assert out.get("parameter.units") == ["K"] * len(target_p)

    actual = out.to_numpy()[:, :2, :2]
    np.testing.assert_allclose(actual, ref_values, atol=1e-8, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("sort_mode", [(None, None), ("ascending", "descending"), ("descending", "ascending")])
@pytest.mark.parametrize(
    "target_h, ref_values",
    [
        (
            [10000.0, -2000.0, 5000.0],
            np.array([
                [[222.80249991296523, 223.3624981324529], [240.17321676165847, 230.05516258566286]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[261.09944396779997, 257.4088919626445], [273.74894013310427, 266.7762235885782]],
            ]),
        ),
    ],
)
def test_fieldlist_grib_interpolate_hybrid_to_height_levels_core(sort_mode, target_h, ref_values):
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tq_ml137.grib2")

    sp = ds.sel({"parameter.variable": "lnsp", "vertical.level": 1})[0]
    sp = sp.set(values=np.exp(sp.values))

    t = ds.sel({"parameter.variable": "t"})
    if sort_mode[0] is not None:
        t = t.order_by({"vertical.level": sort_mode[0]})

    q = ds.sel({"parameter.variable": "q"})
    if sort_mode[1] is not None:
        q = q.order_by({"vertical.level": sort_mode[1]})

    zs = ds.sel({"parameter.variable": "z", "vertical.level": 1})[0]

    out = vertical.interpolate_hybrid_to_height_levels(t, target_h, t, q, zs, sp)

    assert isinstance(out, FieldList)
    assert len(out) == len(target_h)
    assert out.get("vertical.level_type") == ["height"] * len(target_h)
    assert out.get("vertical.level") == target_h
    assert out.get("parameter.variable") == ["t"] * len(target_h)
    assert out.get("parameter.units") == ["K"] * len(target_h)

    actual = out.to_numpy()[:, :2, :2]
    np.testing.assert_allclose(actual, ref_values, atol=1e-8, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("sort_mode", [(None, None), ("ascending", "descending"), ("descending", "ascending")])
@pytest.mark.parametrize("aux_type", ["field", "fieldlist", "value"])
@pytest.mark.parametrize(
    "levels, target_h,aux, ref_values",
    [
        (
            None,
            [10000.0, -2000.0, 2.0, 5000.0, 150000.0, 90000.0],
            {"aux_bottom_data": 350, "aux_bottom_h": 0, "aux_top_data": 10, "aux_top_h": 100000.0},
            np.array([
                [[222.80249991296523, 223.3624981324529], [240.17321676165847, 230.05516258566286]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[336.7809892845, 337.5907916552], [340.1367262347, 339.6298627815]],
                [[261.09944396779997, 257.4088919626445], [273.74894013310427, 266.7762235885782]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[112.6181184463, 113.4113699832], [108.0271163263, 102.0249230605]],
            ]),
        ),
        (
            list(range(90, 138)),
            [-2000.0, 2.0, 5000.0, 20000.0, 15000.0],
            {"aux_bottom_data": 350, "aux_bottom_h": 0, "aux_top_data": 10, "aux_top_h": 16000.0},
            np.array([
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[336.7809892845, 337.5907916552], [340.1367262347, 339.6298627815]],
                [[261.09944396779997, 257.4088919626445], [273.74894013310427, 266.7762235885782]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[37.0246775043, 36.1995290430], [39.6086104236, 36.8273890951]],
            ]),
        ),
    ],
)
def test_fieldlist_grib_interpolate_hybrid_to_height_levels_aux(sort_mode, aux_type, levels, target_h, aux, ref_values):
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tq_ml137.grib2")

    sp = ds.sel({"parameter.variable": "lnsp", "vertical.level": 1})[0]
    sp = sp.set(values=np.exp(sp.values))

    t = ds.sel({"parameter.variable": "t"})
    if sort_mode[0] is not None:
        t = t.order_by({"vertical.level": sort_mode[0]})

    q = ds.sel({"parameter.variable": "q"})
    if sort_mode[1] is not None:
        q = q.order_by({"vertical.level": sort_mode[1]})

    zs = ds.sel({"parameter.variable": "z", "vertical.level": 1})[0]

    if levels is not None:
        r = []
        for f in t:
            if f.get("vertical.level") in levels:
                r.append(f)
        t = FieldList.from_fields(r)
        r = []
        for f in q:
            if f.get("vertical.level") in levels:
                r.append(f)
        q = FieldList.from_fields(r)

    aux = dict(aux)
    if aux_type == "fieldlist":
        for key, value in aux.items():
            aux[key] = FieldList.from_fields([sp.set(values=np.full_like(sp.values, value))])
    elif aux_type == "field":
        for key, value in aux.items():
            aux[key] = sp.set(values=np.full_like(sp.values, value))
    elif aux_type != "value":
        raise ValueError(f"Unsupported aux_type: {aux_type}")

    out = vertical.interpolate_hybrid_to_height_levels(t, target_h, t, q, zs, sp, **aux)

    assert isinstance(out, FieldList)
    assert len(out) == len(target_h)
    assert out.get("vertical.level_type") == ["height"] * len(target_h)
    assert out.get("vertical.level") == target_h
    assert out.get("parameter.variable") == ["t"] * len(target_h)
    assert out.get("parameter.units") == ["K"] * len(target_h)

    actual = out.to_numpy()[:, :2, :2]
    np.testing.assert_allclose(actual, ref_values, atol=1e-8, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("sort_mode", [(None, None), ("ascending", "descending"), ("descending", "ascending")])
@pytest.mark.parametrize(
    "target_h, ref_values",
    [
        (
            [10000.0, -2000.0, 5000.0],
            np.array([
                [[228.258349516436, 228.258349516436], [228.66183646999937, 228.652909342945]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[255.19437602531926, 255.19437602531926], [258.4087786252097, 258.59473571612006]],
            ]),
        ),
    ],
)
def test_fieldlist_grib_interpolate_pressure_to_height_levels_core(sort_mode, target_h, ref_values):
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tz_pl.grib1")

    zs = ds.sel({"parameter.variable": "z", "vertical.level_type": "surface"})[0]

    t = ds.sel({"parameter.variable": "t", "vertical.level_type": "pressure"})
    if sort_mode[0] is not None:
        t = t.order_by({"vertical.level": sort_mode[0]})

    z = ds.sel({"parameter.variable": "z", "vertical.level_type": "pressure"})
    if sort_mode[1] is not None:
        z = z.order_by({"vertical.level": sort_mode[1]})

    out = vertical.interpolate_pressure_to_height_levels(t, target_h, z, zs=zs)

    assert isinstance(out, FieldList)
    assert len(out) == len(target_h)
    assert out.get("vertical.level_type") == ["height"] * len(target_h)
    assert out.get("vertical.level") == target_h
    assert out.get("parameter.variable") == ["t"] * len(target_h)
    assert out.get("parameter.units") == ["K"] * len(target_h)

    actual = out.to_numpy()[:, :2, :2]
    np.testing.assert_allclose(actual, ref_values, atol=1e-8, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("sort_mode", [(None, None), ("ascending", "descending"), ("descending", "ascending")])
@pytest.mark.parametrize("aux_type", ["field", "fieldlist", "value"])
@pytest.mark.parametrize(
    "target_h,aux, ref_values",
    [
        (
            [10000.0, -2000.0, -800, 50000.0, 20000.0],
            {"aux_bottom_data": 350, "aux_bottom_h": -1000, "aux_top_data": 10, "aux_top_h": 25000.0},
            np.array([
                [[228.258349516436, 228.258349516436], [228.66183646999937, 228.652909342945]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[335.0779071966, 335.0779071966], [336.0075765093, 336.1604932347]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[135.4588853459, 135.4588853459], [135.2427489047, 135.3647594209]],
            ]),
        ),
    ],
)
def test_fieldlist_grib_interpolate_pressure_to_height_levels_aux(sort_mode, aux_type, target_h, aux, ref_values):
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tz_pl.grib1")

    zs = ds.sel({"parameter.variable": "z", "vertical.level_type": "surface"})[0]

    t = ds.sel({"parameter.variable": "t", "vertical.level_type": "pressure"})
    if sort_mode[0] is not None:
        t = t.order_by({"vertical.level": sort_mode[0]})

    z = ds.sel({"parameter.variable": "z", "vertical.level_type": "pressure"})
    if sort_mode[1] is not None:
        z = z.order_by({"vertical.level": sort_mode[1]})

    aux = dict(aux)
    if aux_type == "fieldlist":
        for key, value in aux.items():
            aux[key] = FieldList.from_fields([t[0].set(values=np.full_like(t[0].values, value))])
    elif aux_type == "field":
        for key, value in aux.items():
            aux[key] = t[0].set(values=np.full_like(t[0].values, value))
    elif aux_type != "value":
        raise ValueError(f"Unsupported aux_type: {aux_type}")

    out = vertical.interpolate_pressure_to_height_levels(t, target_h, z, zs=zs, **aux)

    assert isinstance(out, FieldList)
    assert len(out) == len(target_h)
    assert out.get("vertical.level_type") == ["height"] * len(target_h)
    assert out.get("vertical.level") == target_h
    assert out.get("parameter.variable") == ["t"] * len(target_h)
    assert out.get("parameter.units") == ["K"] * len(target_h)

    actual = out.to_numpy()[:, :2, :2]
    np.testing.assert_allclose(actual, ref_values, atol=1e-8, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("sort_mode", [("ascending", "descending")])
@pytest.mark.parametrize(
    "target_coord, ref_values",
    [
        (
            [50000.0, 1000000.0, 85000.0],
            np.array([
                [[252.625839233398438, 252.625839233398438], [255.506698608398438, 255.624862670898438]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[271.075210571289062, 271.075210571289062], [274.143569946289062, 273.864273071289062]],
            ]),
        ),
        (
            50000.0,
            np.array([
                [[252.625839233398438, 252.625839233398438], [255.506698608398438, 255.624862670898438]],
            ]),
        ),
        (
            [50000.0],
            np.array([
                [[252.625839233398438, 252.625839233398438], [255.506698608398438, 255.624862670898438]],
            ]),
        ),
    ],
)
def test_fieldlist_grib_interpolate_monotonic_pl_to_pl_scalar(sort_mode, target_coord, ref_values):
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tz_pl.grib1")

    t = ds.sel({"parameter.variable": "t", "vertical.level_type": "pressure"})
    if sort_mode[0] is not None:
        t = t.order_by({"vertical.level": sort_mode[0]})

    out = vertical.interpolate_monotonic(t, coord=None, target_coord=target_coord, coord_type="pressure")

    if isinstance(target_coord, (int, float)):
        target_coord = [target_coord]

    assert isinstance(out, FieldList)
    assert len(out) == len(target_coord)
    assert out.get("vertical.level_type") == ["pressure"] * len(target_coord)
    assert np.allclose(np.array(out.get("vertical.level")) * 100, np.array(target_coord))
    assert out.get("parameter.variable") == ["t"] * len(target_coord)
    assert out.get("parameter.units") == ["K"] * len(target_coord)

    actual = out.to_numpy()[:, :2, :2]
    np.testing.assert_allclose(actual, ref_values, atol=1e-8, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("sort_mode", ["ascending", "descending"])
@pytest.mark.parametrize(
    "target_index,ref_levels,ref_values",
    [
        (
            0,
            np.array([502.71716247558595]),
            np.array([[[252.8275077819, 252.8275077819], [255.7030556266, 255.8207520890]]]),
        ),
        (
            [0],
            np.array([502.71716247558595]),
            np.array([[[252.8275077819, 252.8275077819], [255.7030556266, 255.8207520890]]]),
        ),
        (
            [0, 1],
            np.array([502.71716247558595, 502.7107521057129]),
            np.array([
                [[252.8275077819, 252.8275077819], [255.7030556266, 255.8207520890]],
                [[252.8270320025, 252.8270320025], [255.7024048907, 255.8200620057]],
            ]),
        ),
    ],
)
def test_fieldlist_grib_interpolate_monotonic_pl_to_pl_field(sort_mode, target_index, ref_levels, ref_values):
    from earthkit.data import Field, FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tz_pl.grib1")

    t = ds.sel({"parameter.variable": "t", "vertical.level_type": "pressure"})

    # create target pressure field/fieldlist buy "perturbing"  50000 Pa
    # with some temperature values
    if isinstance(target_index, int):
        target_coord = t[target_index] + 50000.0  # create a Field for target_coord with the same metadata as t
    elif isinstance(target_index, list):
        target_coord = FieldList.from_fields([t[idx] + 50000.0 for idx in target_index])
    else:
        raise ValueError("Invalid target_index type")

    if sort_mode is not None:
        t = t.order_by({"vertical.level": sort_mode})

    out = vertical.interpolate_monotonic(t, coord=None, target_coord=target_coord, coord_type="pressure")

    if isinstance(target_coord, Field):
        target_coord = FieldList.from_fields([target_coord])

    assert isinstance(out, FieldList)
    assert len(out) == len(target_coord)
    assert out.get("vertical.level_type") == ["pressure"] * len(target_coord)
    assert np.allclose(np.array(out.get("vertical.level")), ref_levels)
    assert out.get("parameter.variable") == ["t"] * len(target_coord)
    assert out.get("parameter.units") == ["K"] * len(target_coord)

    actual = out.to_numpy()[:, :2, :2]
    np.testing.assert_allclose(actual, ref_values, atol=1e-8, rtol=1e-6, equal_nan=True)
