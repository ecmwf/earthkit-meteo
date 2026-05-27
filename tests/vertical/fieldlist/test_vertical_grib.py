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
def test_fieldlist_grib_interpolate_hybrid_to_pressure_levels(sort_mode):
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tq_ml137.grib2")

    sp = ds.sel({"parameter.variable": "lnsp", "vertical.level": 1})[0]
    sp = sp.set(values=np.exp(sp.values))

    t = ds.sel({"parameter.variable": "t"})
    if sort_mode is not None:
        t = t.order_by({"vertical.level": sort_mode})

    target_p = [50000.0, 1000000.0, 85000.0]

    out = vertical.interpolate_hybrid_to_pressure_levels(t, target_p, sp)

    assert isinstance(out, FieldList)
    assert len(out) == len(target_p)
    assert out.get("vertical.level_type") == ["pressure"] * len(out)
    assert out.get("vertical.level") == target_p
    assert out.get("parameter.variable") == ["t"] * len(target_p)
    assert out.get("parameter.units") == ["K"] * len(target_p)

    # 50000 Pa (500 hPa)
    np.testing.assert_allclose(
        out[0].to_numpy()[:2, :2],
        [[256.84623176888516, 253.75744555161938], [268.90648252133036, 268.83196698107815]],
        atol=1e-8,
        rtol=1e-6,
    )

    # 1000000 Pa (10 hPa) is above the top of the atmosphere, so all values are NaN
    assert np.all(np.isnan(out[1].to_numpy()[:2, :2]))

    # 85000 Pa (850 hPa)
    np.testing.assert_allclose(
        out[2].to_numpy()[:2, :2],
        [[279.2465996835962, 277.5093980166771], [291.93205593367657, 294.0204548739655]],
        atol=1e-8,
        rtol=1e-6,
    )


@pytest.mark.parametrize("sort_mode", [(None, None)])
def test_fieldlist_grib_interpolate_hybrid_to_height_levels(sort_mode):
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

    target_h = [10000.0, -2000.0, 5000.0]

    out = vertical.interpolate_hybrid_to_height_levels(t, target_h, t, q, zs, sp)

    assert isinstance(out, FieldList)
    assert len(out) == len(target_h)
    assert out.get("vertical.level_type") == ["height"] * len(target_h)
    assert out.get("vertical.level") == target_h
    assert out.get("parameter.variable") == ["t"] * len(target_h)
    assert out.get("parameter.units") == ["K"] * len(target_h)

    # 10000 m
    np.testing.assert_allclose(
        out[0].to_numpy()[:2, :2],
        [[222.80249991296523, 223.3624981324529], [240.17321676165847, 230.05516258566286]],
        atol=1e-8,
        rtol=1e-6,
    )

    # -2000 m is below the surface, so all values are NaN
    assert np.all(np.isnan(out[1].to_numpy()[:2, :2]))

    # 5000 m
    np.testing.assert_allclose(
        out[2].to_numpy()[:2, :2],
        [[261.09944396779997, 257.4088919626445], [273.74894013310427, 266.7762235885782]],
        atol=1e-8,
        rtol=1e-6,
    )
