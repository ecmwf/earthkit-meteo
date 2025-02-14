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
from earthkit.meteo.utils.testing import ARRAY_BACKENDS

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
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
def test_wind_speed(u, v, v_ref, array_backend):
    u, v, v_ref = array_backend.asarray(u, v, v_ref)
    sp = wind.speed(u, v)
    assert array_backend.allclose(sp, v_ref, equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
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
def test_wind_direction(u, v, v_ref, kwargs, array_backend):
    u, v, v_ref = array_backend.asarray(u, v, v_ref)
    d = wind.direction(u, v, **kwargs)
    assert array_backend.allclose(d, v_ref, equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
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
def test_wind_xy_to_polar(u, v, sp_ref, d_ref, array_backend):
    u, v, sp_ref, d_ref = array_backend.asarray(u, v, sp_ref, d_ref)
    sp, d = wind.xy_to_polar(u, v)
    assert array_backend.allclose(sp, sp_ref, equal_nan=True)
    assert array_backend.allclose(d, d_ref, equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
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
def test_wind_polar_to_xy(sp, d, u_ref, v_ref, array_backend):
    sp, d, u_ref, v_ref = array_backend.asarray(sp, d, u_ref, v_ref)
    u, v = wind.polar_to_xy(sp, d)
    assert array_backend.allclose(u, u_ref, equal_nan=True, atol=1e-7)
    assert array_backend.allclose(v, v_ref, equal_nan=True, atol=1e-7)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "omega,t,p,v_ref", [([1.2, 21.3], [285.6, 261.1], [1000, 850], [-0.1003208031, -1.9152219066])]
)
def test_w_from_omega(omega, t, p, v_ref, array_backend):
    omega, t, p, v_ref = array_backend.asarray(omega, t, p, v_ref)
    p = p * 100.0
    w = wind.w_from_omega(omega, t, p)
    assert array_backend.allclose(w, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("lat, v_ref", [([-20, 0, 50], [-0.0000498810, 0.0, 0.0001117217])])
def test_coriolis(lat, v_ref, array_backend):
    lat, v_ref = array_backend.asarray(lat, v_ref)
    c = wind.coriolis(lat)
    assert array_backend.allclose(c, v_ref, rtol=1e-04)


# histogram2d is not available in torch, so we skip this test for now
@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "sp,d,sectors,sp_bins,percent,v_ref,dir_bin_ref",
    [
        (
            [3.5, 1, 1.1, 2.1, 0.1, 0.0, 2.4, 1.9, 1.7, 3.9, 3.1, 2.1, np.nan, np.nan],
            [1.0, 29, 31, 93.0, 121, 171, 189, 245, 240.11, 311, 359.1, np.nan, 11, np.nan],
            6,
            [0, 1, 2, 3, 4],
            False,
            [
                [0.0000000000, 0.0000000000, 1.0000000000, 1.0000000000, 0.0000000000, 0.0000000000],
                [1.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 2.0000000000, 0.0000000000],
                [0.0000000000, 0.0000000000, 1.0000000000, 1.0000000000, 0.0000000000, 0.0000000000],
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
            [1.0, 29, 31, 93.0, 121, 171, 189, 245, 240.11, 311, 359.1, np.nan, 11, np.nan],
            6,
            [0, 1, 2, 3, 4],
            True,
            np.array(
                [
                    [0.0000000000, 0.0000000000, 1.0000000000, 1.0000000000, 0.0000000000, 0.0000000000],
                    [1.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 2.0000000000, 0.0000000000],
                    [0.0000000000, 0.0000000000, 1.0000000000, 1.0000000000, 0.0000000000, 0.0000000000],
                    [
                        2.0000000000,
                        0.0000000000,
                        0.0000000000,
                        0.0000000000,
                        0.0000000000,
                        1.0000000000,
                    ],
                ]
            )
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
        (3.4, 90.01, 1, [0, 5], False, [[1]], [-180.0000000000, 180.0000000000]),
    ],
)
def test_windrose_1(sp, d, sectors, sp_bins, percent, v_ref, dir_bin_ref, array_backend):
    sp, d, sp_bins, v_ref, dir_bin_ref = array_backend.asarray(sp, d, sp_bins, v_ref, dir_bin_ref)

    dir_bin_ref = array_backend.namespace.astype(dir_bin_ref, sp.dtype)

    r = wind.windrose(sp, d, sectors=sectors, speed_bins=sp_bins, percent=percent)

    dir_bin_ref = array_backend.astype(dir_bin_ref, r[0].dtype)
    v_ref = array_backend.astype(v_ref, r[1].dtype)

    assert array_backend.allclose(r[0], v_ref, equal_nan=True, rtol=1e-04)
    assert array_backend.allclose(r[1], dir_bin_ref, equal_nan=True, rtol=1e-04)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "sp,d,sectors,sp_bins", [(3.4, 90.01, 0, [0, 1]), (3.4, 90.01, 6, [0]), (3.4, 90.01, 6, None)]
)
def test_windrose_invalid(sp, d, sectors, sp_bins, array_backend):
    if sp_bins is not None:
        sp_bins = array_backend.asarray(sp_bins)
    sp, d = array_backend.asarray(sp, d)

    with pytest.raises(ValueError):
        wind.windrose(sp, d, sectors=sectors, speed_bins=sp_bins, percent=False)
