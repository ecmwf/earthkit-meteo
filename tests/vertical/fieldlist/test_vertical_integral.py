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

from earthkit.meteo.utils import testing
from earthkit.meteo.utils.testing import NO_EKD

np.set_printoptions(formatter={"float_kind": "{:.15f}".format})
pytestmark = pytest.mark.skipif(NO_EKD, reason="EKD is not installed")

SHAPE = (2, 3)


def _get_fieldlist(name):
    import earthkit.data as ekd

    path = testing.get_test_data(name, "test-data")
    return ekd.from_source("file", path).to_fieldlist()


def _field(values, variable, units, level, level_type, member=None, base_datetime=None, step=None):
    from earthkit.data import Field

    kwargs = {}
    if member is not None:
        kwargs["ensemble"] = {"member": member}
    if base_datetime is not None or step is not None:
        time = {}
        if base_datetime is not None:
            time["base_datetime"] = base_datetime
        if step is not None:
            time["step"] = step
        kwargs["time"] = time

    return Field.from_components(
        values=np.asarray(values, dtype=float),
        parameter={"variable": variable, "units": units},
        vertical={"level": level, "level_type": level_type},
        **kwargs,
    )


def _column(values, levels, variable="q", units="K", level_type="height_above_ground_level", **kwargs):
    r"""Build a column FieldList: one field per level, with the same parameter and metadata."""
    from earthkit.data import FieldList

    return FieldList.from_fields([
        _field(v, variable, units, level, level_type, **kwargs) for v, level in zip(values, levels)
    ])


def _profile(values, levels, level_type, variable="pres", units="Pa", shape=SHAPE):
    r"""Build a profile FieldList: one field per level, each of them constant over the grid."""
    from earthkit.data import FieldList

    values = np.asarray(values, dtype=float)[:, None, None] * np.ones(shape)
    return FieldList.from_fields([_field(v, variable, units, level, level_type) for v, level in zip(values, levels)])


def _check_integral(out, variable, units, shape=None, template=None):
    from earthkit.data import FieldList
    from earthkit.utils.units import Units

    assert isinstance(out, FieldList)
    assert len(out) == 1

    field = out[0]
    assert field.get("parameter.variable") == variable
    assert field.get("vertical.level_type") == "entire_atmosphere"
    assert field.get("vertical.level") == 0
    assert Units.from_any(field.get("parameter.units")) == Units.from_any(units)
    if shape is not None:
        assert field.shape == shape

    if template is not None:
        # apart from the vertical coordinate and the units everything is copied
        assert field.get("parameter.long_name") == template.get("parameter.long_name")
        assert field.get("time.base_datetime") == template.get("time.base_datetime")
        assert field.get("time.step") == template.get("time.step")
        assert field.get("geography.grid_type") == template.get("geography.grid_type")


# --------------------------------------------------------------------------------------
# vertical_integral_pressure
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sort_mode", [None, "ascending", "descending"])
def test_fieldlist_grib_vertical_integral_pressure_from_levels(sort_mode):
    import earthkit.meteo.vertical.array as array
    import earthkit.meteo.vertical.fieldlist as vertical
    from earthkit.meteo.utils.fieldlist import pressure_from_metadata

    fl = _get_fieldlist("tuv_pl.grib").sel({"parameter.variable": "t", "vertical.level_type": "pressure"})
    ref = fl.order_by({"vertical.level": "ascending"})
    if sort_mode is not None:
        fl = fl.order_by({"vertical.level": sort_mode})

    # the pressure is taken from the level metadata: hPa converted to Pa
    p = np.asarray(pressure_from_metadata(ref), dtype=float)
    np.testing.assert_allclose(p, [30000.0, 40000.0, 50000.0, 70000.0, 85000.0, 100000.0])
    expected = array.vertical_integral_pressure(ref.to_numpy(copy=False), p, vertical_dim=0)

    out = vertical.vertical_integral_pressure(fl)

    _check_integral(out, "t", "K kg m-2", ref[0].shape, template=ref[0])
    np.testing.assert_allclose(out.to_numpy(copy=False)[0], expected, atol=1e-8, rtol=1e-6)


def test_fieldlist_grib_vertical_integral_pressure_p_from_fieldlist_same_levels():
    import earthkit.meteo.vertical.array as array
    import earthkit.meteo.vertical.fieldlist as vertical
    from earthkit.meteo.utils.fieldlist import pressure_from_metadata

    ds = _get_fieldlist("tuv_pl.grib")
    t = ds.sel({"parameter.variable": "t", "vertical.level_type": "pressure"})
    ref = t.order_by({"vertical.level": "ascending"})
    levels = np.asarray(ref.get("vertical.level"), dtype=float)

    p = _profile(levels * 100.0, levels, "pressure", shape=ref[0].shape)
    expected = array.vertical_integral_pressure(ref.to_numpy(copy=False), p.to_numpy(copy=False), vertical_dim=0)

    # both the data and the pressure profile are given in a shuffled order
    out = vertical.vertical_integral_pressure(ref[::-1], p=p[::-1])

    _check_integral(out, "t", "K kg m-2", ref[0].shape)
    np.testing.assert_allclose(out.to_numpy(copy=False)[0], expected, atol=1e-8, rtol=1e-6)

    # and it is the same as the result computed for the sorted input
    ordered = vertical.vertical_integral_pressure(ref, p=p)
    np.testing.assert_allclose(ordered.to_numpy(copy=False)[0], out.to_numpy(copy=False)[0], atol=1e-12, rtol=1e-12)

    # the pressure values are the ones of the metadata
    metadata_out = vertical.vertical_integral_pressure(ref)
    np.testing.assert_allclose(metadata_out.to_numpy(copy=False)[0], out.to_numpy(copy=False)[0], atol=1e-8, rtol=1e-6)
    assert np.allclose(np.asarray(pressure_from_metadata(ref), dtype=float), levels * 100.0)


@pytest.mark.parametrize("sort_mode", [None, "ascending", "descending"])
def test_fieldlist_grib_vertical_integral_pressure_hybrid_interfaces(sort_mode):
    import earthkit.meteo.vertical.array as array
    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tq_ml137.grib2")
    sp = ds.sel({"parameter.variable": "lnsp", "vertical.level": 1})[0]
    sp = sp.set(values=np.exp(sp.values))

    # N + 1 layer interfaces: the hybrid half-levels 0, ..., N
    p_half = vertical.pressure_on_hybrid_levels(sp, alpha_top="ifs", output="half")

    t = ds.sel({"parameter.variable": "t"})
    ref = t.order_by({"vertical.level": "ascending"})
    if sort_mode is not None:
        t = t.order_by({"vertical.level": sort_mode})

    assert len(p_half) == len(ref) + 1

    expected = array.vertical_integral_pressure(ref.to_numpy(copy=False), p_half.to_numpy(copy=False), vertical_dim=0)

    out = vertical.vertical_integral_pressure(t, p=p_half)

    _check_integral(out, "t", "K kg m-2", ref[0].shape, template=ref[0])
    np.testing.assert_allclose(out.to_numpy(copy=False)[0], expected, atol=1e-8, rtol=1e-6)


def test_fieldlist_grib_vertical_integral_pressure_hybrid_levels():
    import earthkit.meteo.vertical.array as array
    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tq_ml137.grib2")
    sp = ds.sel({"parameter.variable": "lnsp", "vertical.level": 1})[0]
    sp = sp.set(values=np.exp(sp.values))

    # N fields: the hybrid full-levels, i.e. the trapezoidal rule
    p_full = vertical.pressure_on_hybrid_levels(sp, alpha_top="ifs", output="full")
    t = ds.sel({"parameter.variable": "t"}).order_by({"vertical.level": "ascending"})

    expected = array.vertical_integral_pressure(t.to_numpy(copy=False), p_full.to_numpy(copy=False), vertical_dim=0)

    out = vertical.vertical_integral_pressure(t, p=p_full)

    _check_integral(out, "t", "K kg m-2", t[0].shape)
    np.testing.assert_allclose(out.to_numpy(copy=False)[0], expected, atol=1e-8, rtol=1e-6)

    # the interfaces and the full-levels define different columns
    p_half = vertical.pressure_on_hybrid_levels(sp, alpha_top="ifs", output="half")
    other = vertical.vertical_integral_pressure(t, p=p_half)
    assert not np.allclose(out.to_numpy(copy=False)[0], other.to_numpy(copy=False)[0])


def test_fieldlist_vertical_integral_pressure_field_input():
    import earthkit.meteo.vertical.array as array
    import earthkit.meteo.vertical.fieldlist as vertical

    # a single field with the two interfaces enclosing it is a valid column
    data = _column([np.arange(6, dtype=float).reshape(SHAPE)], [1], level_type="hybrid")
    p = _profile([50000.0, 100000.0], [0, 1], "hybrid")

    expected = array.vertical_integral_pressure(data.to_numpy(copy=False), p.to_numpy(copy=False), vertical_dim=0)

    out = vertical.vertical_integral_pressure(data[0], p=p)

    _check_integral(out, "q", "K kg m-2", SHAPE)
    np.testing.assert_allclose(out.to_numpy(copy=False)[0], expected, atol=1e-8, rtol=1e-6)


def test_fieldlist_vertical_integral_pressure_interfaces_of_several_parameters():
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.array as array
    import earthkit.meteo.vertical.fieldlist as vertical

    levels = [1, 2, 3]
    data = _column([np.full(SHAPE, 1.0 + i) for i in range(len(levels))], levels, level_type="hybrid")

    # the interfaces may be built from several parameters: the surface pressure closes the column
    p = FieldList.from_fields([
        _field(np.full(SHAPE, 80000.0), "sp", "Pa", 0, "hybrid"),
        *_profile([70000.0, 50000.0, 30000.0], levels, "hybrid"),
    ])

    expected = array.vertical_integral_pressure(data.to_numpy(copy=False), p.to_numpy(copy=False), vertical_dim=0)

    out = vertical.vertical_integral_pressure(data, p=p)

    _check_integral(out, "q", "K kg m-2", SHAPE)
    np.testing.assert_allclose(out.to_numpy(copy=False)[0], expected, atol=1e-8, rtol=1e-6)


# --------------------------------------------------------------------------------------
# vertical_integral_height
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("level_type", ["height_above_ground_level", "height_above_mean_sea_level"])
@pytest.mark.parametrize("sort_mode", [None, "ascending", "descending"])
def test_fieldlist_vertical_integral_height_from_levels(level_type, sort_mode):
    import earthkit.meteo.vertical.array as array
    import earthkit.meteo.vertical.fieldlist as vertical

    levels = [0.0, 500.0, 1500.0, 5000.0]
    values = [np.full(SHAPE, 10.0 + 100.0 * lev) for lev in levels]
    data = _column(values, levels, level_type=level_type)
    ref = data.order_by({"vertical.level": "ascending"})
    if sort_mode is not None:
        data = data.order_by({"vertical.level": sort_mode})

    # the height is taken from the level metadata (m)
    expected = array.vertical_integral_height(ref.to_numpy(copy=False), np.asarray(levels, dtype=float), vertical_dim=0)

    out = vertical.vertical_integral_height(data)

    _check_integral(out, "q", "K m", SHAPE)
    np.testing.assert_allclose(out.to_numpy(copy=False)[0], expected, atol=1e-8, rtol=1e-6)


def test_fieldlist_vertical_integral_height_h_fieldlist_and_rho():
    import earthkit.meteo.vertical.array as array
    import earthkit.meteo.vertical.fieldlist as vertical

    levels = [0.0, 500.0, 1500.0]
    values = [np.full(SHAPE, 10.0 + 100.0 * lev) for lev in levels]
    data = _column(values, levels)

    # N + 1 interfaces enclosing the data levels
    h_levels = [-100.0, 250.0, 1000.0, 5000.0]
    h = _profile(h_levels, h_levels, "height_above_ground_level", variable="h", units="m")
    rho = _profile([1.0, 0.9, 0.8], levels, "height_above_ground_level", variable="rho", units="kg m-3")

    expected = array.vertical_integral_height(
        data.to_numpy(copy=False), h.to_numpy(copy=False), rho=rho.to_numpy(copy=False), vertical_dim=0
    )

    out = vertical.vertical_integral_height(data[::-1], h=h[::-1], rho=rho[::-1])

    _check_integral(out, "q", "K kg m-2", SHAPE)
    np.testing.assert_allclose(out.to_numpy(copy=False)[0], expected, atol=1e-8, rtol=1e-6)


def test_fieldlist_vertical_integral_height_h_fieldlist_same_levels():
    import earthkit.meteo.vertical.array as array
    import earthkit.meteo.vertical.fieldlist as vertical

    levels = [0.0, 500.0, 1500.0]
    values = [np.full(SHAPE, 10.0 + 100.0 * lev) for lev in levels]
    data = _column(values, levels)

    h = _profile(levels, levels, "height_above_ground_level", variable="h", units="m")

    expected = array.vertical_integral_height(data.to_numpy(copy=False), h.to_numpy(copy=False), vertical_dim=0)

    out = vertical.vertical_integral_height(data, h=h)

    _check_integral(out, "q", "K m", SHAPE)
    np.testing.assert_allclose(out.to_numpy(copy=False)[0], expected, atol=1e-8, rtol=1e-6)
    # the levels of the data are the same as the ones of the profile
    np.testing.assert_allclose(out.to_numpy(copy=False)[0], vertical.vertical_integral_height(data).to_numpy()[0])


def test_fieldlist_vertical_integral_height_rho_on_other_level_type():
    import earthkit.meteo.vertical.array as array
    import earthkit.meteo.vertical.fieldlist as vertical

    levels = [0.0, 500.0, 1500.0]
    values = [np.full(SHAPE, 10.0 + 100.0 * lev) for lev in levels]
    data = _column(values, levels)

    # the density does not need to be on the same level type as the data, only the levels match
    rho = _profile([1.0, 0.9, 0.8], levels, "hybrid", variable="rho", units="kg m-3")

    expected = array.vertical_integral_height(
        data.to_numpy(copy=False), np.asarray(levels, dtype=float), rho=rho.to_numpy(copy=False), vertical_dim=0
    )

    out = vertical.vertical_integral_height(data, rho=rho)

    _check_integral(out, "q", "K kg m-2", SHAPE)
    np.testing.assert_allclose(out.to_numpy(copy=False)[0], expected, atol=1e-8, rtol=1e-6)


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


def test_fieldlist_vertical_integral_several_parameters():
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    ds = _get_fieldlist("tuv_pl.grib")
    t = ds.sel({"parameter.variable": "t", "vertical.level_type": "pressure"})
    u = ds.sel({"parameter.variable": "u", "vertical.level_type": "pressure"})

    # two parameters on two different levels
    fl = FieldList.from_fields([t[0], u[1]])

    with pytest.raises(ValueError, match='single vertical column.*"parameter.variable"'):
        vertical.vertical_integral_pressure(fl)

    with pytest.raises(ValueError, match='single vertical column.*"parameter.variable"'):
        vertical.vertical_integral_height(fl.order_by({"vertical.level": "descending"}))


@pytest.mark.parametrize("field_kwargs", [{"base_datetime": "2024-01-02T12:00:00"}, {"step": 6}, {"member": 2}])
def test_fieldlist_vertical_integral_not_a_single_column(field_kwargs):
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    levels = [0.0, 500.0, 1500.0]
    values = [np.full(SHAPE, 10.0 + 100.0 * lev) for lev in levels]
    kwargs = {"base_datetime": "2024-01-01T12:00:00", "step": 0, "member": 1}
    data = _column(values, levels, **kwargs)

    other = _field(values[-1], "q", "K", levels[-1], "height_above_ground_level", **{**kwargs, **field_kwargs})
    mixed = FieldList.from_fields([data[0], data[1], other])

    with pytest.raises(ValueError, match="single vertical column"):
        vertical.vertical_integral_height(mixed)


def test_fieldlist_vertical_integral_duplicate_levels():
    import earthkit.meteo.vertical.fieldlist as vertical

    data = _column([np.zeros(SHAPE), np.ones(SHAPE)], [0.0, 0.0])

    with pytest.raises(ValueError, match="one field for each vertical level"):
        vertical.vertical_integral_height(data)


def test_fieldlist_vertical_integral_mixed_level_types():
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    mixed = FieldList.from_fields([
        _field(np.zeros(SHAPE), "q", "K", 0.0, "height_above_ground_level"),
        _field(np.ones(SHAPE), "q", "K", 1, "hybrid"),
    ])

    with pytest.raises(ValueError, match="single vertical level type"):
        vertical.vertical_integral_height(mixed)


def test_fieldlist_vertical_integral_pressure_no_p_on_hybrid_levels():
    import earthkit.meteo.vertical.fieldlist as vertical

    t = _get_fieldlist("tq_ml137.grib2").sel({"parameter.variable": "t"})

    with pytest.raises(ValueError, match="must be on pressure levels"):
        vertical.vertical_integral_pressure(t)


def test_fieldlist_vertical_integral_height_no_h_on_pressure_levels():
    import earthkit.meteo.vertical.fieldlist as vertical

    t = _get_fieldlist("tuv_pl.grib").sel({"parameter.variable": "t", "vertical.level_type": "pressure"})

    with pytest.raises(ValueError, match="height_above_ground_level or height_above_mean_sea_level"):
        vertical.vertical_integral_height(t)


@pytest.mark.parametrize(
    "p_levels,p_values",
    [
        ([1.0], [100000.0]),
        ([1.0, 2.0], [100000.0, 90000.0]),
        ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [100000.0] * 9),
    ],
)
def test_fieldlist_vertical_integral_pressure_wrong_number_of_p_fields(p_levels, p_values):
    import earthkit.meteo.vertical.fieldlist as vertical

    data = _column([np.zeros(SHAPE) + i for i in range(3)], [1.0, 2.0, 3.0], level_type="hybrid")
    p = _profile(p_values, p_levels, "hybrid")

    with pytest.raises(ValueError, match=r'"p" must contain 3 \(on the data levels\) or 4 \(on the layer interfaces\)'):
        vertical.vertical_integral_pressure(data, p=p)


def test_fieldlist_vertical_integral_pressure_p_levels_do_not_match():
    import earthkit.meteo.vertical.fieldlist as vertical

    data = _column([np.zeros(SHAPE) + i for i in range(3)], [1.0, 2.0, 3.0], level_type="hybrid")
    p = _profile([100000.0, 90000.0, 80000.0], [1.0, 2.0, 4.0], "hybrid")

    with pytest.raises(ValueError, match=r'Levels of "p" and "data" do not match'):
        vertical.vertical_integral_pressure(data, p=p)


def test_fieldlist_vertical_integral_height_rho_levels_do_not_match():
    import earthkit.meteo.vertical.fieldlist as vertical

    data = _column([np.zeros(SHAPE) + i for i in range(3)], [0.0, 500.0, 1500.0])
    rho = _profile([1.0, 0.9, 0.8], [0.0, 500.0, 1000.0], "hybrid", variable="rho", units="kg m-3")

    with pytest.raises(ValueError, match=r'Levels of "rho" and "data" do not match'):
        vertical.vertical_integral_height(data, rho=rho)


def test_fieldlist_vertical_integral_rho_on_interfaces():
    import earthkit.meteo.vertical.fieldlist as vertical

    data = _column([np.zeros(SHAPE) + i for i in range(3)], [0.0, 500.0, 1500.0])

    # rho is only allowed on the data levels
    rho = _profile([1.0, 0.9, 0.8, 0.7], [-100.0, 250.0, 1000.0, 5000.0], "hybrid", variable="rho", units="kg m-3")

    with pytest.raises(ValueError, match=r'"rho" must contain 3 fields'):
        vertical.vertical_integral_height(data, rho=rho)


def test_fieldlist_vertical_integral_not_a_fieldlist():
    from earthkit.data import FieldList

    import earthkit.meteo.vertical.fieldlist as vertical

    with pytest.raises(ValueError, match="must be a FieldList"):
        vertical.vertical_integral_pressure(np.zeros((3, *SHAPE)))

    with pytest.raises(ValueError, match="must be a FieldList"):
        vertical.vertical_integral_height(np.zeros((3, *SHAPE)))

    with pytest.raises(ValueError, match="at least one field"):
        vertical.vertical_integral_height(FieldList.from_fields([]))

    data = _column([np.zeros(SHAPE) + i for i in range(3)], [0.0, 500.0, 1500.0])
    with pytest.raises(ValueError, match="must be a FieldList"):
        vertical.vertical_integral_height(data, h=[0.0, 500.0, 1500.0])


def test_fieldlist_vertical_integral_single_level_needs_interfaces():
    import earthkit.meteo.vertical.fieldlist as vertical

    data = _column([np.zeros(SHAPE)], [1.0], level_type="hybrid")

    # a single level cannot be integrated with the trapezoidal rule
    with pytest.raises(ValueError, match="At least two levels"):
        vertical.vertical_integral_pressure(data, p=_profile([100000.0], [1.0], "hybrid"))


# --------------------------------------------------------------------------------------
# units of the result
# --------------------------------------------------------------------------------------


def _assert_units(field, expected):
    """Check the parameter units of the result field against ``expected``."""
    from earthkit.utils.units import Units

    assert Units.from_any(field.get("parameter.units")) == Units.from_any(expected)


def test_fieldlist_vertical_integral_units_unknown_input():
    import earthkit.meteo.vertical.fieldlist as vertical

    levels = [0.0, 500.0, 1500.0]
    values = [np.full(SHAPE, 10.0 + 100.0 * lev) for lev in levels]

    # "unknown" is how earthkit-data marks the units it cannot resolve. The units of the
    # result stay unknown: they are neither kept as they were nor mixed with the ones of
    # the integral, which would give a meaningless "unknown*kg*m^-2".
    data = _column(values, levels, units="unknown")

    _assert_units(vertical.vertical_integral_height(data)[0], "unknown")
    rho = _profile([1.0, 0.9, 0.8], levels, "hybrid", variable="rho")
    _assert_units(vertical.vertical_integral_height(data, rho=rho)[0], "unknown")

    p = _profile((np.asarray(levels) * 100.0).tolist(), levels, "hybrid", variable="pres", units="Pa")
    _assert_units(vertical.vertical_integral_pressure(data, p=p)[0], "unknown")

    # the values do not depend on the units
    out = vertical.vertical_integral_height(data)
    ref = vertical.vertical_integral_height(_column(values, levels))
    np.testing.assert_allclose(out.to_numpy(copy=False), ref.to_numpy(copy=False), rtol=1e-12)


def test_fieldlist_vertical_integral_units_missing():
    from earthkit.utils.units import Units

    import earthkit.meteo.vertical.fieldlist as vertical

    levels = [0.0, 500.0, 1500.0]
    values = [np.full(SHAPE, 10.0 + 100.0 * lev) for lev in levels]

    # earthkit-data parses the missing units of a field as dimensionless, so the units of
    # the result are the ones of the integral alone and never the ones of the input
    data = _column(values, levels, units=None)
    assert Units.from_any(data[0].get("parameter.units")) == Units.from_any(None)

    _assert_units(vertical.vertical_integral_height(data)[0], "m")

    rho = _profile([1.0, 0.9, 0.8], levels, "hybrid", variable="rho", units="kg m-3")
    _assert_units(vertical.vertical_integral_height(data, rho=rho)[0], "kg m-2")


# --------------------------------------------------------------------------------------
# the top-level, dispatching function
# --------------------------------------------------------------------------------------


def test_high_level_vertical_integral_pressure():
    import earthkit.meteo.vertical as vertical_high_level
    import earthkit.meteo.vertical.fieldlist as vertical

    fl = _get_fieldlist("tuv_pl.grib").sel({"parameter.variable": "t", "vertical.level_type": "pressure"})

    out = vertical_high_level.vertical_integral_pressure(fl)
    ref = vertical.vertical_integral_pressure(fl)

    _check_integral(out, "t", "K kg m-2", fl[0].shape)
    np.testing.assert_allclose(out.to_numpy(copy=False), ref.to_numpy(copy=False), atol=1e-12, rtol=1e-12)

    # "vertical_dim" is not supported for FieldList input
    with pytest.raises(TypeError, match="vertical_integral_pressure"):
        vertical_high_level.vertical_integral_pressure(fl, vertical_dim=0)


def test_high_level_vertical_integral_height():
    import earthkit.meteo.vertical as vertical_high_level
    import earthkit.meteo.vertical.fieldlist as vertical

    levels = [0.0, 500.0, 1500.0]
    values = [np.full(SHAPE, 10.0 + 100.0 * lev) for lev in levels]
    data = _column(values, levels)
    rho = _profile([1.0, 0.9, 0.8], levels, "hybrid", variable="rho", units="kg m-3")

    out = vertical_high_level.vertical_integral_height(data, rho=rho)
    ref = vertical.vertical_integral_height(data, rho=rho)

    _check_integral(out, "q", "K kg m-2", SHAPE)
    np.testing.assert_allclose(out.to_numpy(copy=False), ref.to_numpy(copy=False), atol=1e-12, rtol=1e-12)
