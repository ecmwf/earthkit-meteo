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
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo import wind

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
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
def test_wind_speed(u, v, v_ref, xp, device):
    u = xp.asarray(u, device=device)
    v = xp.asarray(v, device=device)
    v_ref = xp.asarray(v_ref, device=device)
    sp = wind.speed(u, v)
    assert xp.allclose(sp, v_ref, equal_nan=True)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
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
def test_wind_direction(u, v, v_ref, kwargs, xp, device):
    u = xp.asarray(u, device=device)
    v = xp.asarray(v, device=device)
    v_ref = xp.asarray(v_ref, device=device)
    d = wind.direction(u, v, **kwargs)
    assert xp.allclose(d, v_ref, equal_nan=True)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
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
def test_wind_xy_to_polar(u, v, sp_ref, d_ref, xp, device):
    u = xp.asarray(u, device=device)
    v = xp.asarray(v, device=device)
    sp_ref = xp.asarray(sp_ref, device=device)
    d_ref = xp.asarray(d_ref, device=device)
    sp, d = wind.xy_to_polar(u, v)
    assert xp.allclose(sp, sp_ref, equal_nan=True)
    assert xp.allclose(d, d_ref, equal_nan=True)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
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
def test_wind_polar_to_xy(sp, d, u_ref, v_ref, xp, device):
    u_ref = xp.asarray(u_ref, device=device)
    d = xp.asarray(d, device=device)
    sp = xp.asarray(sp, device=device)
    v_ref = xp.asarray(v_ref, device=device)
    u, v = wind.polar_to_xy(sp, d)
    assert xp.allclose(u, u_ref, equal_nan=True, atol=1e-7)
    assert xp.allclose(v, v_ref, equal_nan=True, atol=1e-7)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "omega,t,p,v_ref", [([1.2, 21.3], [285.6, 261.1], [1000, 850], [-0.1003208031, -1.9152219066])]
)
def test_w_from_omega(omega, t, p, v_ref, xp, device):
    v_ref = xp.asarray(v_ref, device=device)
    omega = xp.asarray(omega, device=device)
    t = xp.asarray(t, device=device)
    p = xp.asarray(p, device=device)
    p = p * 100.0
    w = wind.w_from_omega(omega, t, p)
    assert xp.allclose(w, v_ref)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("lat, v_ref", [([-20, 0, 50], [-0.0000498810, 0.0, 0.0001117217])])
def test_coriolis(lat, v_ref, xp, device):
    v_ref = xp.asarray(v_ref, device=device)
    lat = xp.asarray(lat, device=device)
    c = wind.coriolis(lat)
    assert xp.allclose(c, v_ref, rtol=1e-04)


# histogram2d is not available in torch, so we skip this test for now
@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
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
def test_windrose_1(sp, d, sectors, sp_bins, percent, v_ref, dir_bin_ref, xp, device):
    sp, d, sp_bins, v_ref, dir_bin_ref = (
        xp.asarray(x, device=device) for x in [sp, d, sp_bins, v_ref, dir_bin_ref]
    )

    dir_bin_ref = xp.astype(dir_bin_ref, sp.dtype)

    r = wind.windrose(sp, d, sectors=sectors, speed_bins=sp_bins, percent=percent)

    dir_bin_ref = xp.astype(dir_bin_ref, r[0].dtype)
    v_ref = xp.astype(v_ref, r[1].dtype)

    assert xp.allclose(r[0], v_ref, equal_nan=True, rtol=1e-04)
    assert xp.allclose(r[1], dir_bin_ref, equal_nan=True, rtol=1e-04)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "sp,d,sectors,sp_bins", [(3.4, 90.01, 0, [0, 1]), (3.4, 90.01, 6, [0]), (3.4, 90.01, 6, None)]
)
def test_windrose_invalid(sp, d, sectors, sp_bins, xp, device):
    if sp_bins is not None:
        sp_bins = xp.asarray(sp_bins)
    sp, d = xp.asarray(sp, device=device), xp.asarray(d, device=device)

    with pytest.raises(ValueError):
        wind.windrose(sp, d, sectors=sectors, speed_bins=sp_bins, percent=False)
