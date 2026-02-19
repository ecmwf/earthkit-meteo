# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import os

import numpy as np
import pytest

from earthkit.meteo import thermo
from earthkit.meteo.utils.testing import NO_XARRAY

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})
pytestmark = pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")


def _da(x):
    import xarray as xr

    return xr.DataArray(np.asarray(x))


def _scalar_da(x):
    import xarray as xr

    return xr.DataArray(np.asarray(x))


def _np(x):
    return np.asarray(x)


def _xr_da_1d(x, dim="point"):
    import xarray as xr

    return xr.DataArray(np.asarray(x), dims=(dim,))


def _data_file(name):
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", name)


def _read_data_file(path):
    import numpy as np

    d = np.genfromtxt(
        _data_file(path),
        delimiter=",",
        names=True,
    )
    return d


# ---------------------------------------------------------------------
# Refs are regenerated from thermo.array (current implementation)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "t_c",
    [
        [-273.15, -40.0, 0.0, 20.0, 100.0, np.nan],
        0.0,
    ],
)
def test_xr_celsius_to_kelvin(t_c):
    t_da = _scalar_da(t_c) if np.isscalar(t_c) else _da(t_c)
    out = thermo.celsius_to_kelvin(t_da)
    ref = thermo.array.celsius_to_kelvin(_np(t_c))
    assert np.allclose(out.values, ref, equal_nan=True)
    if np.isscalar(t_c):
        assert out.ndim == 0


@pytest.mark.parametrize(
    "t_k",
    [
        [0.0, 233.15, 273.15, 293.15, 373.15, np.nan],
        273.16,
    ],
)
def test_xr_kelvin_to_celsius(t_k):
    t_da = _scalar_da(t_k) if np.isscalar(t_k) else _da(t_k)
    out = thermo.kelvin_to_celsius(t_da)
    ref = thermo.array.kelvin_to_celsius(_np(t_k))
    assert np.allclose(out.values, ref, equal_nan=True)
    if np.isscalar(t_k):
        assert out.ndim == 0


@pytest.mark.parametrize(
    "q,p",
    [
        ([0.01], [100000.0]),
        (0.01, 100000.0),
    ],
)
def test_xr_vapour_pressure_from_specific_humidity(q, p):
    q_da = _scalar_da(q) if np.isscalar(q) else _da(q)
    p_da = _scalar_da(p) if np.isscalar(p) else _da(p)
    out = thermo.vapour_pressure_from_specific_humidity(q_da, p_da)
    ref = thermo.array.vapour_pressure_from_specific_humidity(_np(q), _np(p))
    assert np.allclose(out.values, ref, equal_nan=True)
    if np.isscalar(q) and np.isscalar(p):
        assert out.ndim == 0


@pytest.mark.parametrize(
    "w,p",
    [
        ([0.01], [100000.0]),
        (0.01, 100000.0),
    ],
)
def test_xr_vapour_pressure_from_mixing_ratio(w, p):
    w_da = _scalar_da(w) if np.isscalar(w) else _da(w)
    p_da = _scalar_da(p) if np.isscalar(p) else _da(p)
    out = thermo.vapour_pressure_from_mixing_ratio(w_da, p_da)
    ref = thermo.array.vapour_pressure_from_mixing_ratio(_np(w), _np(p))
    assert np.allclose(out.values, ref, equal_nan=True)
    if np.isscalar(w) and np.isscalar(p):
        assert out.ndim == 0


@pytest.mark.parametrize(
    "e,p",
    [
        ([1000.0], [100000.0]),
        (1000.0, 100000.0),
        (1000, 100000),  # int inputs (catches dtype/NaN issues)
    ],
)
def test_xr_specific_humidity_from_vapour_pressure(e, p):
    e_da = _scalar_da(e) if np.isscalar(e) else _da(e)
    p_da = _scalar_da(p) if np.isscalar(p) else _da(p)
    out = thermo.specific_humidity_from_vapour_pressure(e_da, p_da)
    ref = thermo.array.specific_humidity_from_vapour_pressure(_np(e), _np(p))
    assert np.allclose(out.values, ref, equal_nan=True)
    if np.isscalar(e) and np.isscalar(p):
        assert out.ndim == 0


@pytest.mark.parametrize(
    "e,p",
    [
        ([1000.0], [100000.0]),
        (1000.0, 100000.0),
        (1000, 100000),
    ],
)
def test_xr_mixing_ratio_from_vapour_pressure(e, p):
    e_da = _scalar_da(e) if np.isscalar(e) else _da(e)
    p_da = _scalar_da(p) if np.isscalar(p) else _da(p)
    out = thermo.mixing_ratio_from_vapour_pressure(e_da, p_da)
    ref = thermo.array.mixing_ratio_from_vapour_pressure(_np(e), _np(p))
    assert np.allclose(out.values, ref, equal_nan=True)
    if np.isscalar(e) and np.isscalar(p):
        assert out.ndim == 0


@pytest.mark.parametrize(
    "t,phase",
    [
        (273.15, "water"),
        (273.15, "ice"),
        (273.15, "mixed"),
    ],
)
def test_xr_saturation_vapour_pressure(t, phase):
    t_da = _scalar_da(t)
    out = thermo.saturation_vapour_pressure(t_da, phase=phase)
    ref = thermo.array.saturation_vapour_pressure(_np(t), phase=phase)
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,p,phase",
    [
        (273.15, 100000.0, "water"),
    ],
)
def test_xr_saturation_mixing_ratio(t, p, phase):
    t_da = _scalar_da(t)
    p_da = _scalar_da(p)
    out = thermo.saturation_mixing_ratio(t_da, p_da, phase=phase)
    ref = thermo.array.saturation_mixing_ratio(_np(t), _np(p), phase=phase)
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,p,phase",
    [
        (273.15, 100000.0, "water"),
    ],
)
def test_xr_saturation_specific_humidity(t, p, phase):
    t_da = _scalar_da(t)
    p_da = _scalar_da(p)
    out = thermo.saturation_specific_humidity(t_da, p_da, phase=phase)
    ref = thermo.array.saturation_specific_humidity(_np(t), _np(p), phase=phase)
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,phase",
    [
        (273.15, "water"),
    ],
)
def test_xr_saturation_vapour_pressure_slope(t, phase):
    t_da = _scalar_da(t)
    out = thermo.saturation_vapour_pressure_slope(t_da, phase=phase)
    ref = thermo.array.saturation_vapour_pressure_slope(_np(t), phase=phase)
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "es",
    [
        611.21,
    ],
)
def test_xr_temperature_from_saturation_vapour_pressure(es):
    es_da = _scalar_da(es)
    out = thermo.temperature_from_saturation_vapour_pressure(es_da)
    ref = thermo.array.temperature_from_saturation_vapour_pressure(_np(es))
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,td",
    [
        (293.15, 283.15),
    ],
)
def test_xr_relative_humidity_from_dewpoint(t, td):
    t_da = _scalar_da(t)
    td_da = _scalar_da(td)
    out = thermo.relative_humidity_from_dewpoint(t_da, td_da)
    ref = thermo.array.relative_humidity_from_dewpoint(_np(t), _np(td))
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,q,p",
    [
        (293.15, 0.01, 100000.0),
    ],
)
def test_xr_relative_humidity_from_specific_humidity(t, q, p):
    t_da = _scalar_da(t)
    q_da = _scalar_da(q)
    p_da = _scalar_da(p)
    out = thermo.relative_humidity_from_specific_humidity(t_da, q_da, p_da)
    ref = thermo.array.relative_humidity_from_specific_humidity(_np(t), _np(q), _np(p))
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "td,p",
    [
        (283.15, 100000.0),
    ],
)
def test_xr_specific_humidity_from_dewpoint(td, p):
    td_da = _scalar_da(td)
    p_da = _scalar_da(p)
    out = thermo.specific_humidity_from_dewpoint(td_da, p_da)
    ref = thermo.array.specific_humidity_from_dewpoint(_np(td), _np(p))
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "td,p",
    [
        (283.15, 100000.0),
    ],
)
def test_xr_mixing_ratio_from_dewpoint(td, p):
    td_da = _scalar_da(td)
    p_da = _scalar_da(p)
    out = thermo.mixing_ratio_from_dewpoint(td_da, p_da)
    ref = thermo.array.mixing_ratio_from_dewpoint(_np(td), _np(p))
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,r,p",
    [
        (293.15, 70.0, 100000.0),
    ],
)
def test_xr_specific_humidity_from_relative_humidity(t, r, p):
    t_da = _scalar_da(t)
    r_da = _scalar_da(r)
    p_da = _scalar_da(p)
    out = thermo.specific_humidity_from_relative_humidity(t_da, r_da, p_da)
    ref = thermo.array.specific_humidity_from_relative_humidity(_np(t), _np(r), _np(p))
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,r",
    [
        (293.15, 70.0),
    ],
)
def test_xr_dewpoint_from_relative_humidity(t, r):
    t_da = _scalar_da(t)
    r_da = _scalar_da(r)
    out = thermo.dewpoint_from_relative_humidity(t_da, r_da)
    ref = thermo.array.dewpoint_from_relative_humidity(_np(t), _np(r))
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "q,p",
    [
        (0.01, 100000.0),
    ],
)
def test_xr_dewpoint_from_specific_humidity(q, p):
    q_da = _scalar_da(q)
    p_da = _scalar_da(p)
    out = thermo.dewpoint_from_specific_humidity(q_da, p_da)
    ref = thermo.array.dewpoint_from_specific_humidity(_np(q), _np(p))
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,q",
    [
        (293.15, 0.01),
    ],
)
def test_xr_virtual_temperature(t, q):
    t_da = _scalar_da(t)
    q_da = _scalar_da(q)
    out = thermo.virtual_temperature(t_da, q_da)
    ref = thermo.array.virtual_temperature(_np(t), _np(q))
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,q,p",
    [
        (293.15, 0.01, 100000.0),
    ],
)
def test_xr_virtual_potential_temperature(t, q, p):
    t_da = _scalar_da(t)
    q_da = _scalar_da(q)
    p_da = _scalar_da(p)
    out = thermo.virtual_potential_temperature(t_da, q_da, p_da)
    ref = thermo.array.virtual_potential_temperature(_np(t), _np(q), _np(p))
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,p",
    [
        (293.15, 100000.0),
    ],
)
def test_xr_potential_temperature(t, p):
    t_da = _scalar_da(t)
    p_da = _scalar_da(p)
    out = thermo.potential_temperature(t_da, p_da)
    ref = thermo.array.potential_temperature(_np(t), _np(p))
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "theta,p",
    [
        (320.0, 85000.0),
    ],
)
def test_xr_temperature_from_potential_temperature(theta, p):
    theta_da = _scalar_da(theta)
    p_da = _scalar_da(p)
    out = thermo.temperature_from_potential_temperature(theta_da, p_da)
    ref = thermo.array.temperature_from_potential_temperature(_np(theta), _np(p))
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,t_def,p_def",
    [
        (283.15, 293.15, 100000.0),
    ],
)
def test_xr_pressure_on_dry_adiabat(t, t_def, p_def):
    t_da = _scalar_da(t)
    out = thermo.pressure_on_dry_adiabat(t_da, t_def=t_def, p_def=p_def)
    ref = thermo.array.pressure_on_dry_adiabat(_np(t), t_def=t_def, p_def=p_def)
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "p,t_def,p_def",
    [
        (85000.0, 293.15, 100000.0),
    ],
)
def test_xr_temperature_on_dry_adiabat(p, t_def, p_def):
    p_da = _scalar_da(p)
    out = thermo.temperature_on_dry_adiabat(p_da, t_def=t_def, p_def=p_def)
    ref = thermo.array.temperature_on_dry_adiabat(_np(p), t_def=t_def, p_def=p_def)
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,td,method",
    [
        (293.15, 283.15, "davies"),
        (293.15, 283.15, "bolton"),
    ],
)
def test_xr_lcl_temperature(t, td, method):
    t_da = _scalar_da(t)
    td_da = _scalar_da(td)
    out = thermo.lcl_temperature(t_da, td_da, method=method)
    ref = thermo.array.lcl_temperature(_np(t), _np(td), method=method)
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,td,p,method",
    [
        (293.15, 283.15, 100000.0, "davies"),
    ],
)
def test_xr_lcl(t, td, p, method):
    t_da = _scalar_da(t)
    td_da = _scalar_da(td)
    p_da = _scalar_da(p)
    t_lcl, p_lcl = thermo.lcl(t_da, td_da, p_da, method=method)
    t_ref, p_ref = thermo.array.lcl(_np(t), _np(td), _np(p), method=method)
    assert np.allclose(t_lcl.values, t_ref, equal_nan=True)
    assert np.allclose(p_lcl.values, p_ref, equal_nan=True)
    assert t_lcl.ndim == 0 and p_lcl.ndim == 0


@pytest.mark.parametrize(
    "t,td,p,method",
    [
        (293.15, 283.15, 100000.0, "ifs"),
        (293.15, 283.15, 100000.0, "bolton35"),
        (293.15, 283.15, 100000.0, "bolton39"),
    ],
)
def test_xr_ept_from_dewpoint(t, td, p, method):
    t_da = _scalar_da(t)
    td_da = _scalar_da(td)
    p_da = _scalar_da(p)
    out = thermo.ept_from_dewpoint(t_da, td_da, p_da, method=method)
    ref = thermo.array.ept_from_dewpoint(_np(t), _np(td), _np(p), method=method)
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,q,p,method",
    [
        (293.15, 0.01, 100000.0, "ifs"),
    ],
)
def test_xr_ept_from_specific_humidity(t, q, p, method):
    t_da = _scalar_da(t)
    q_da = _scalar_da(q)
    p_da = _scalar_da(p)
    out = thermo.ept_from_specific_humidity(t_da, q_da, p_da, method=method)
    ref = thermo.array.ept_from_specific_humidity(_np(t), _np(q), _np(p), method=method)
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,p,method",
    [
        (293.15, 100000.0, "ifs"),
    ],
)
def test_xr_saturation_ept(t, p, method):
    t_da = _scalar_da(t)
    p_da = _scalar_da(p)
    out = thermo.saturation_ept(t_da, p_da, method=method)
    ref = thermo.array.saturation_ept(_np(t), _np(p), method=method)
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize("ept_method", ["ifs", "bolton35", "bolton39"])
@pytest.mark.parametrize("t_method", ["bisect", "newton"])
def test_xr_temperature_on_moist_adiabat(ept_method, t_method):
    # read input vectors from the same ref file as the array test
    ref_file = "t_on_most_adiabat.csv"
    ref = _read_data_file(ref_file)

    ept = _xr_da_1d(ref["ept"])
    p = _xr_da_1d(ref["p"])

    out = thermo.temperature_on_moist_adiabat(ept, p, ept_method=ept_method, t_method=t_method)

    # reference: current numpy/array implementation
    v_ref = thermo.array.temperature_on_moist_adiabat(
        np.asarray(ref["ept"]),
        np.asarray(ref["p"]),
        ept_method=ept_method,
        t_method=t_method,
    )

    assert np.allclose(out.values, v_ref, equal_nan=True), f"{ept_method=} {t_method=}"
    assert out.dims == ("point",)


@pytest.mark.parametrize("ept_method", ["ifs", "bolton35", "bolton39"])
@pytest.mark.parametrize("t_method", ["bisect", "newton"])
def test_xr_wet_bulb_temperature_from_dewpoint_vectorized(ept_method, t_method):
    t = _xr_da_1d([293.15, 300.0])
    td = _xr_da_1d([283.15, 290.0])
    p = _xr_da_1d([100000.0, 90000.0])

    out = thermo.wet_bulb_temperature_from_dewpoint(t, td, p, ept_method=ept_method, t_method=t_method)
    ref = thermo.array.wet_bulb_temperature_from_dewpoint(
        np.asarray(t.values),
        np.asarray(td.values),
        np.asarray(p.values),
        ept_method=ept_method,
        t_method=t_method,
    )

    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.dims == ("point",)


@pytest.mark.parametrize(
    "t,q,p,ept_method,t_method",
    [
        ([293.15, 300.0], [0.01, 0.005], [100000.0, 90000.0], "ifs", "bisect"),
    ],
)
def test_xr_wet_bulb_temperature_from_specific_humidity_vectorized(t, q, p, ept_method, t_method):
    t_da = _xr_da_1d(t)
    q_da = _xr_da_1d(q)
    p_da = _xr_da_1d(p)

    out = thermo.wet_bulb_temperature_from_specific_humidity(
        t_da, q_da, p_da, ept_method=ept_method, t_method=t_method
    )

    ref = thermo.array.wet_bulb_temperature_from_specific_humidity(
        np.asarray(t), np.asarray(q), np.asarray(p), ept_method=ept_method, t_method=t_method
    )

    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.dims == ("point",)


@pytest.mark.parametrize(
    "t, td, p, ept_method, t_method",
    [
        (293.15, 283.15, 100000.0, "ifs", "direct"),
    ],
)
def test_xr_wet_bulb_potential_temperature_from_dewpoint(t, td, p, ept_method, t_method):
    t_da = _scalar_da(t)
    td_da = _scalar_da(td)
    p_da = _scalar_da(p)

    out = thermo.wet_bulb_potential_temperature_from_dewpoint(
        t_da, td_da, p_da, ept_method=ept_method, t_method=t_method
    )
    ref = thermo.array.wet_bulb_potential_temperature_from_dewpoint(
        _np(t), _np(td), _np(p), ept_method=ept_method, t_method=t_method
    )

    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize(
    "t,q,p,method",
    [
        (293.15, 0.01, 100000.0, "ifs"),
    ],
)
def test_xr_wet_bulb_potential_temperature_from_specific_humidity(t, q, p, method):
    t_da = _scalar_da(t)
    q_da = _scalar_da(q)
    p_da = _scalar_da(p)

    out = thermo.wet_bulb_potential_temperature_from_specific_humidity(t_da, q_da, p_da, ept_method=method)
    ref = thermo.array.wet_bulb_potential_temperature_from_specific_humidity(
        _np(t), _np(q), _np(p), ept_method=method
    )

    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0


@pytest.mark.parametrize("q", [0.01])
def test_xr_specific_gas_constant(q):
    q_da = _scalar_da(q)
    out = thermo.specific_gas_constant(q_da)
    ref = thermo.array.specific_gas_constant(_np(q))
    assert np.allclose(out.values, ref, equal_nan=True)
    assert out.ndim == 0
