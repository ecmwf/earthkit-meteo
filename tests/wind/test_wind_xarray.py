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

from earthkit.meteo import wind
from earthkit.meteo.utils.testing import NO_XARRAY

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})

pytestmark = pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")


def _da(values):
    import xarray as xr

    return xr.DataArray(np.asarray(values))


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
    u, v = wind.polar_to_xy(_da(sp), _da(d))
    assert np.allclose(u.values, np.asarray(u_ref), equal_nan=True, atol=1e-7)
    assert np.allclose(v.values, np.asarray(v_ref), equal_nan=True, atol=1e-7)


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize(
    "omega,t,p,v_ref", [([1.2, 21.3], [285.6, 261.1], [1000, 850], [-0.1003208031, -1.9152219066])]
)
def test_w_from_omega(omega, t, p, v_ref):
    omega = _da(omega)
    t = _da(t)
    p = _da(p) * 100.0
    w = wind.w_from_omega(omega, t, p)
    assert np.allclose(w.values, np.asarray(v_ref))


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize("lat, v_ref", [([-20, 0, 50], [-0.0000498810, 0.0, 0.0001117217])])
def test_xr_coriolis(lat, v_ref):
    c = wind.coriolis(_da(lat))
    assert np.allclose(c.values, np.asarray(v_ref), rtol=1e-04)


# @pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
# @pytest.mark.parametrize(
#     "sp,d,sectors,sp_bins,percent,v_ref,dir_bin_ref",
#     [
#         (
#             [3.5, 1, 1.1, 2.1, 0.1, 0.0, 2.4, 1.9, 1.7, 3.9, 3.1, 2.1, np.nan, np.nan],
#             [1.0, 29, 31, 93.0, 121, 171, 189, 245, 240.11, 311, 359.1, np.nan, 11, np.nan],
#             6,
#             [0, 1, 2, 3, 4],
#             False,
#             [
#                 [0.0000000000, 0.0000000000, 1.0000000000, 1.0000000000, 0.0000000000, 0.0000000000],
#                 [1.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 2.0000000000, 0.0000000000],
#                 [0.0000000000, 0.0000000000, 1.0000000000, 1.0000000000, 0.0000000000, 0.0000000000],
#                 [
#                     2.0000000000,
#                     0.0000000000,
#                     0.0000000000,
#                     0.0000000000,
#                     0.0000000000,
#                     1.0000000000,
#                 ],
#             ],
#             [
#                 -30.0000000000,
#                 30.0000000000,
#                 90.0000000000,
#                 150.0000000000,
#                 210.0000000000,
#                 270.0000000000,
#                 330.0000000000,
#             ],
#         ),
#         (
#             [3.5, 1, 1.1, 2.1, 0.1, 0.0, 2.4, 1.9, 1.7, 3.9, 3.1, 2.1, np.nan, np.nan],
#             [1.0, 29, 31, 93.0, 121, 171, 189, 245, 240.11, 311, 359.1, np.nan, 11, np.nan],
#             6,
#             [0, 1, 2, 3, 4],
#             True,
#             np.array(
#                 [
#                     [0.0000000000, 0.0000000000, 1.0000000000, 1.0000000000, 0.0000000000, 0.0000000000],
#                     [1.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 2.0000000000, 0.0000000000],
#                     [0.0000000000, 0.0000000000, 1.0000000000, 1.0000000000, 0.0000000000, 0.0000000000],
#                     [
#                         2.0000000000,
#                         0.0000000000,
#                         0.0000000000,
#                         0.0000000000,
#                         0.0000000000,
#                         1.0000000000,
#                     ],
#                 ]
#             )
#             * 100
#             / 11.0,
#             [
#                 -30.0000000000,
#                 30.0000000000,
#                 90.0000000000,
#                 150.0000000000,
#                 210.0000000000,
#                 270.0000000000,
#                 330.0000000000,
#             ],
#         ),
#         (
#             3.4,
#             90.01,
#             6,
#             [0, 5],
#             False,
#             [[0, 0, 1, 0, 0, 0]],
#             [
#                 -30.0000000000,
#                 30.0000000000,
#                 90.0000000000,
#                 150.0000000000,
#                 210.0000000000,
#                 270.0000000000,
#                 330.0000000000,
#             ],
#         ),
#         (3.4, 90.01, 1, [0, 5], False, [[1]], [-180.0000000000, 180.0000000000]),
#     ],
# )
# def test_xr_windrose_1(sp, d, sectors, sp_bins, percent, v_ref, dir_bin_ref):
#     r = wind.windrose(_da(sp), _da(d), sectors=sectors, speed_bins=sp_bins, percent=percent)
#     assert np.allclose(r[0].values, np.asarray(v_ref), equal_nan=True, rtol=1e-04)
#     assert np.allclose(r[1].values, np.asarray(dir_bin_ref), equal_nan=True, rtol=1e-04)


# @pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
# @pytest.mark.parametrize(
#     "sp,d,sectors,sp_bins", [(3.4, 90.01, 0, [0, 1]), (3.4, 90.01, 6, [0]), (3.4, 90.01, 6, None)]
# )
# def test_xr_windrose_invalid(sp, d, sectors, sp_bins):
#     with pytest.raises(ValueError):
#         wind.windrose(_da(sp), _da(d), sectors=sectors, speed_bins=sp_bins, percent=False)
