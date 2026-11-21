# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import earthkit.meteo.wind.array as array_wind
import earthkit.meteo.wind.xarray as wind
from earthkit.meteo.utils.testing import NO_EKD, NO_XARRAY

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})

pytestmark = pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")


def _da(values, dtype=None):
    import xarray as xr

    return xr.DataArray(np.asarray(values, dtype=dtype))


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize(
    "u,v,v_ref",
    [
        (
            [0, 1, 1, 1, 0, -1, -1, -1, 0, np.nan, 1, np.nan],
            [1, 1, 0, -1, -1, -1, 0, 1, 0, 1, np.nan, np.nan],
            [
                1.0,
                1.4142135624,
                1.0,
                1.4142135624,
                1.0,
                1.4142135624,
                1.0,
                1.4142135624,
                0.0,
                np.nan,
                np.nan,
                np.nan,
            ],
        )
    ],
)
def test_xr_wind_speed(u, v, v_ref):
    sp = wind.speed(_da(u), _da(v))
    assert np.allclose(sp.values, np.asarray(v_ref), equal_nan=True)


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize(
    "u,v,kwargs,v_ref",
    [
        (
            [0, 1, 1, 1, 0, -1, -1, -1, 0, np.nan, 1, np.nan],
            [1, 1, 0, -1, -1, -1, 0, 1, 0, 1, np.nan, np.nan],
            {"convention": "meteo"},
            [180.0, 225, 270, 315, 0, 45, 90, 135, 270, np.nan, np.nan, np.nan],
        ),
        (
            [0, 1, 1, 1, 0, -1, -1, -1, 0, np.nan, 1, np.nan],
            [1, 1, 0, -1, -1, -1, 0, 1, 0, 1, np.nan, np.nan],
            {"convention": "polar"},
            [
                90,
                45,
                0,
                315,
                270,
                225,
                180,
                135.0,
                0,
                np.nan,
                np.nan,
                np.nan,
            ],
        ),
        (
            [0, 1, 1, 1, 0, -1, -1, -1, 0, np.nan, 1, np.nan],
            [1, 1, 0, -1, -1, -1, 0, 1, 0, 1, np.nan, np.nan],
            {"convention": "polar", "to_positive": False},
            [
                90,
                45,
                0,
                -45,
                -90,
                -135,
                180,
                135,
                0,
                np.nan,
                np.nan,
                np.nan,
            ],
        ),
        (1.0, 1.0, {"convention": "meteo"}, 225.0),
        (1.0, np.nan, {"convention": "meteo"}, np.nan),
        (1.0, 1.0, {"convention": "polar"}, 45.0),
        (1.0, np.nan, {"convention": "polar"}, np.nan),
    ],
)
def test_xr_wind_direction(u, v, v_ref, kwargs):
    d = wind.direction(_da(u), _da(v), **kwargs)
    assert np.allclose(d.values, np.asarray(v_ref), equal_nan=True)


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize(
    "u,v,sp_ref,d_ref",
    [
        (
            [0, 1, 1, 1, 0, -1, -1, -1, 0, np.nan, 1, np.nan],
            [1, 1, 0, -1, -1, -1, 0, 1, 0, 1, np.nan, np.nan],
            [
                1.0,
                1.4142135624,
                1.0,
                1.4142135624,
                1.0,
                1.4142135624,
                1.0,
                1.4142135624,
                0.0,
                np.nan,
                np.nan,
                np.nan,
            ],
            [180.0, 225, 270, 315, 0, 45, 90, 135, 270, np.nan, np.nan, np.nan],
        )
    ],
)
def test_xr_wind_xy_to_polar(u, v, sp_ref, d_ref):
    sp, d = wind.xy_to_polar(_da(u), _da(v))
    assert np.allclose(sp.values, np.asarray(sp_ref), equal_nan=True)
    assert np.allclose(d.values, np.asarray(d_ref), equal_nan=True)


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize(
    "sp,d,u_ref, v_ref",
    [
        (
            [
                1.0,
                1.4142135624,
                1.0,
                1.4142135624,
                1.0,
                1.4142135624,
                1.0,
                1.4142135624,
                0.0,
                np.nan,
                1,
                np.nan,
            ],
            [180.0, 225, 270, 315, 0, 45, 90, 135, 270, 1, np.nan, np.nan],
            [0, 1, 1, 1, 0, -1, -1, -1, 0, np.nan, np.nan, np.nan],
            [1, 1, 0, -1, -1, -1, 0, 1, 0, np.nan, np.nan, np.nan],
        )
    ],
)
def test_xr_wind_polar_to_xy(sp, d, u_ref, v_ref):
    print(_da(sp))
    print(_da(d))
    u, v = wind.polar_to_xy(_da(sp), _da(d))
    assert np.allclose(u.values, np.asarray(u_ref), equal_nan=True, atol=1e-7)
    assert np.allclose(v.values, np.asarray(v_ref), equal_nan=True, atol=1e-7)


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize(
    "omega,t,p,v_ref",
    [([1.2, 21.3], [285.6, 261.1], [1000, 850], [-0.1003208031, -1.9152219066])],
)
def test_xr_w_from_omega_1(omega, t, p, v_ref):
    omega = _da(omega)
    t = _da(t)
    p = _da(p) * 100.0
    w = wind.w_from_omega(omega, t, p)
    assert np.allclose(w.values, np.asarray(v_ref))


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.skipif(NO_EKD, reason="earthkit-data is not installed")
def test_xr_w_from_omega_2():
    import earthkit.data as ekd

    ds = ekd.from_source("sample", "omega_pl.grib").to_xarray()
    omega = ds["w"]
    t = ds["t"]
    p = ds["level"] * 100.0

    w = wind.w_from_omega(omega, t, p)

    w_ref = array_wind.w_from_omega(omega.values, t.values, p.values[:, np.newaxis, np.newaxis])
    assert np.allclose(w.values, np.asarray(w_ref))


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize("lat, v_ref", [([-20, 0, 50], [-0.0000498810, 0.0, 0.0001117217])])
def test_xr_coriolis(lat, v_ref):
    c = wind.coriolis(_da(lat))
    assert np.allclose(c.values, np.asarray(v_ref), rtol=1e-04)


@pytest.mark.parametrize(
    "sp,d,sectors,sp_bins,percent,sectors_start,v_ref,dir_bin_ref",
    [
        (
            [3.5, 1, 1.1, 2.1, 0.1, 0.0, 2.4, 1.9, 1.7, 3.9, 3.1, 2.1, np.nan, np.nan],
            [
                1.0,
                29,
                31,
                93.0,
                121,
                171,
                189,
                245,
                240.11,
                311,
                359.1,
                np.nan,
                11,
                np.nan,
            ],
            6,
            [0, 1, 2, 3, 4],
            False,
            None,
            [
                [
                    0.0000000000,
                    0.0000000000,
                    1.0000000000,
                    1.0000000000,
                    0.0000000000,
                    0.0000000000,
                ],
                [
                    1.0000000000,
                    1.0000000000,
                    0.0000000000,
                    0.0000000000,
                    2.0000000000,
                    0.0000000000,
                ],
                [
                    0.0000000000,
                    0.0000000000,
                    1.0000000000,
                    1.0000000000,
                    0.0000000000,
                    0.0000000000,
                ],
                [
                    2.0000000000,
                    0.0000000000,
                    0.0000000000,
                    0.0000000000,
                    0.0000000000,
                    1.0000000000,
                ],
            ],
            [
                -30.0000000000,
                30.0000000000,
                90.0000000000,
                150.0000000000,
                210.0000000000,
                270.0000000000,
                330.0000000000,
            ],
        ),
        (
            [3.5, 1, 1.1, 2.1, 0.1, 0.0, 2.4, 1.9, 1.7, 3.9, 3.1, 2.1, np.nan, np.nan],
            [
                1.0,
                29,
                31,
                93.0,
                121,
                171,
                189,
                245,
                240.11,
                311,
                359.1,
                np.nan,
                11,
                np.nan,
            ],
            6,
            [0, 1, 2, 3, 4],
            True,
            None,
            np.array([
                [
                    0.0000000000,
                    0.0000000000,
                    1.0000000000,
                    1.0000000000,
                    0.0000000000,
                    0.0000000000,
                ],
                [
                    1.0000000000,
                    1.0000000000,
                    0.0000000000,
                    0.0000000000,
                    2.0000000000,
                    0.0000000000,
                ],
                [
                    0.0000000000,
                    0.0000000000,
                    1.0000000000,
                    1.0000000000,
                    0.0000000000,
                    0.0000000000,
                ],
                [
                    2.0000000000,
                    0.0000000000,
                    0.0000000000,
                    0.0000000000,
                    0.0000000000,
                    1.0000000000,
                ],
            ])
            * 100
            / 11.0,
            [
                -30.0000000000,
                30.0000000000,
                90.0000000000,
                150.0000000000,
                210.0000000000,
                270.0000000000,
                330.0000000000,
            ],
        ),
        (
            3.4,
            90.01,
            6,
            [0, 5],
            False,
            None,
            [[0, 0, 1, 0, 0, 0]],
            [
                -30.0000000000,
                30.0000000000,
                90.0000000000,
                150.0000000000,
                210.0000000000,
                270.0000000000,
                330.0000000000,
            ],
        ),
        (3.4, 90.01, 1, [0, 5], False, None, [[1]], [-180.0000000000, 180.0000000000]),
        (
            [np.nan, np.nan],
            [11.3, 348.6],
            8,
            [0, 10],
            False,
            None,
            np.full((1, 8), np.nan),
            [-22.5 + 45.0 * i for i in range(9)],
        ),
        (
            [1.0, 1.0, 1.0],
            [11.3, 150, 360],
            8,
            [0, 10],
            False,
            0.0,
            [[2, 0, 0, 1, 0, 0, 0, 0]],
            np.linspace(0, 360, 9),
        ),
        (
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [5, 11.3, 150, 350, 360],
            8,
            [0, 10],
            False,
            10,
            [[1, 0, 0, 1, 0, 0, 0, 3]],
            np.linspace(10, 370, 9),
        ),
        (
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [5, 11.3, 150, 350, 360],
            8,
            [0, 10],
            False,
            22.5,
            [[0, 0, 1, 0, 0, 0, 0, 4]],
            np.linspace(22.5, 382.5, 9),
        ),
        (
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [5, 11.3, 150, 350, 360],
            8,
            [0, 10],
            False,
            -50,
            [[1, 3, 0, 0, 1, 0, 0, 0]],
            np.linspace(-50, 310, 9),
        ),
        (
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [5, 11.3, 150, 350, 360],
            8,
            [0, 10],
            False,
            -180,
            [[0, 0, 0, 1, 3, 0, 0, 1]],
            np.linspace(-180, 180, 9),
        ),
        (
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [5, 11.3, 150, 350, 360],
            8,
            [0, 10],
            False,
            179,
            [[0, 0, 0, 1, 3, 0, 0, 1]],
            np.linspace(179, 539, 9),
        ),
        (
            # 7 sectors: sectors_start is -360/14, which is not exactly representable. The
            # sample at the wrap-around (sectors_start + 360) belongs to the first sector.
            [1.0, 1.0, 1.0],
            [334.2857142857143, 100.0, 200.0],
            7,
            [0, 10],
            False,
            None,
            [[1, 0, 1, 0, 1, 0, 0]],
            np.linspace(-360.0 / 14, 360.0 - 360.0 / 14, 8),
        ),
    ],
)
def test_xr_windrose(sp, d, sectors, sp_bins, percent, sectors_start, v_ref, dir_bin_ref):
    hist, dir_bins = wind.windrose(
        _da(sp), _da(d), sectors=sectors, speed_bins=sp_bins, percent=percent, sectors_start=sectors_start
    )

    assert hist.dims == ("speed_bin", "direction_bin")
    assert dir_bins.dims == ("direction_bin_edge",)
    assert np.allclose(hist["speed_bin"].values, np.asarray(sp_bins[:-1]), equal_nan=True, rtol=1e-04)
    assert np.allclose(
        hist["direction_bin"].values,
        np.asarray(dir_bin_ref[:-1]),
        equal_nan=True,
        rtol=1e-04,
    )
    assert np.allclose(hist.values, np.asarray(v_ref), equal_nan=True, rtol=1e-04)
    assert np.allclose(dir_bins.values, np.asarray(dir_bin_ref), equal_nan=True, rtol=1e-04)


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize("sectors", [5, 7, 11, 13, 22])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_xr_windrose_wrap_around(sectors, dtype):
    # 0, 360 and sectors_start + 360 (e.g. 334.2857142857143 for 7 sectors, which is not exactly
    # representable) are the same bearing and must be counted in the first sector instead of
    # being dropped by the wrap-around.
    step = 360.0 / sectors
    start = -step / 2
    sp = [5.0, 5.0, 5.0]
    d = [start + 360.0, 0.0, 360.0]

    hist, dir_bins = wind.windrose(_da(sp, dtype), _da(d, dtype), sectors=sectors, speed_bins=[0, 10], percent=False)

    assert float(hist.sum()) == len(d)
    assert hist.values[0].tolist() == [float(len(d))] + [0.0] * (sectors - 1)
    assert np.allclose(dir_bins.values, np.linspace(start, start + 360.0, sectors + 1))

    pct, _ = wind.windrose(_da(sp, dtype), _da(d, dtype), sectors=sectors, speed_bins=[0, 10], percent=True)
    assert float(pct.sum()) == pytest.approx(100.0, rel=1e-4)

    # the array implementation agrees
    hist_array, _ = array_wind.windrose(
        np.asarray(sp, dtype=dtype),
        np.asarray(d, dtype=dtype),
        sectors=sectors,
        speed_bins=[0, 10],
        percent=False,
    )
    assert np.allclose(hist.values, np.asarray(hist_array))


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize("sectors", [7, 11])
@pytest.mark.parametrize("sp_dtype,d_dtype", [("float32", "float64"), ("float64", "float32")])
def test_xr_windrose_wrap_around_mixed_dtypes(sectors, sp_dtype, d_dtype):
    # The direction bins are built in the dtype of the speed, so the directions are converted to it
    # before they are wrapped. A sample at sectors_start + 360 provided with a wider dtype than the
    # speed must not be dropped.
    step = 360.0 / sectors
    start = -step / 2
    sp = _da([5.0, 5.0, 5.0], sp_dtype)
    d = _da([start + 360.0, 0.0, 360.0], d_dtype)

    hist, dir_bins = wind.windrose(sp, d, sectors=sectors, speed_bins=[0, 10], percent=False)
    assert float(hist.sum()) == 3
    assert np.allclose(dir_bins.values, np.linspace(start, start + 360.0, sectors + 1))

    pct, _ = wind.windrose(sp, d, sectors=sectors, speed_bins=[0, 10], percent=True)
    assert float(pct.sum()) == pytest.approx(100.0, rel=1e-4)

    if sp_dtype == "float32":
        # the bins are float32: 0, 360 and sectors_start + 360 are the same bearing
        assert hist.values[0].tolist() == [3.0] + [0.0] * (sectors - 1)


@pytest.mark.parametrize(
    "sp,d,sectors,sp_bins",
    [(3.4, 90.01, 0, [0, 1]), (3.4, 90.01, 6, [0]), (3.4, 90.01, 6, None)],
)
def test_xr_windrose_invalid(sp, d, sectors, sp_bins):
    with pytest.raises(ValueError):
        wind.windrose(_da(sp), _da(d), sectors=sectors, speed_bins=sp_bins, percent=False)
