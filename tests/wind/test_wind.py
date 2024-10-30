# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np

from earthkit.meteo import wind

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


def test_speed():
    u = np.array([0, 1, 1, 1, 0, -1, -1, -1, 0, np.nan, 1, np.nan])
    v = np.array([1, 1, 0, -1, -1, -1, 0, 1, 0, 1, np.nan, np.nan])
    v_ref = np.array(
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
        ]
    )

    sp = wind.speed(u, v)
    np.testing.assert_allclose(sp, v_ref)


def test_direction():
    # meteo
    u = np.array([0, 1, 1, 1, 0, -1, -1, -1, 0, np.nan, 1, np.nan])
    v = np.array([1, 1, 0, -1, -1, -1, 0, 1, 0, 1, np.nan, np.nan])
    v_ref = np.array([180.0, 225, 270, 315, 0, 45, 90, 135, 270, np.nan, np.nan, np.nan])
    d = wind.direction(u, v, convention="meteo")
    np.testing.assert_allclose(d, v_ref)

    v_ref = np.array(
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
        ]
    )

    # polar
    d = wind.direction(u, v, convention="polar")
    np.testing.assert_allclose(d, v_ref)

    v_ref = np.array(
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
        ]
    )

    d = wind.direction(u, v, convention="polar", to_positive=False)
    np.testing.assert_allclose(d, v_ref)

    # numbers
    u = [1.0, 1]
    v = [1.0, np.nan]
    v_ref = [225, np.nan]
    for i in range(len(u)):
        d = wind.direction(u[i], v[i], convention="meteo")
        np.testing.assert_allclose(d, v_ref[i])

    v_ref = [45, np.nan]
    for i in range(len(u)):
        d = wind.direction(u[i], v[i], convention="polar")
        np.testing.assert_allclose(d, v_ref[i])


def test_xy_to_polar():
    u = np.array([0, 1, 1, 1, 0, -1, -1, -1, 0, np.nan, 1, np.nan])
    v = np.array([1, 1, 0, -1, -1, -1, 0, 1, 0, 1, np.nan, np.nan])
    sp_ref = np.array(
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
        ]
    )
    d_ref = np.array([180.0, 225, 270, 315, 0, 45, 90, 135, 270, np.nan, np.nan, np.nan])
    sp, d = wind.xy_to_polar(u, v)
    np.testing.assert_allclose(sp, sp_ref)
    np.testing.assert_allclose(d, d_ref)


def test_polar_to_xy():
    sp = np.array(
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
        ]
    )
    d = np.array([180.0, 225, 270, 315, 0, 45, 90, 135, 270, 1, np.nan, np.nan])
    u_ref = np.array([0, 1, 1, 1, 0, -1, -1, -1, 0, np.nan, np.nan, np.nan])
    v_ref = np.array([1, 1, 0, -1, -1, -1, 0, 1, 0, np.nan, np.nan, np.nan])
    u, v = wind.polar_to_xy(sp, d)
    np.testing.assert_allclose(u, u_ref, atol=1e-7)
    np.testing.assert_allclose(v, v_ref, atol=1e-7)


def test_w_from_omega():
    omega = np.array([1.2, 21.3])
    t = np.array([285.6, 261.1])
    p = np.array([1000, 850]) * 100
    v_ref = np.array([-0.1003208031, -1.9152219066])
    w = wind.w_from_omega(omega, t, p)
    np.testing.assert_allclose(w, v_ref)


def test_coriolis():
    lat = np.array([-20, 0, 50])
    c = wind.coriolis(lat)
    v_ref = np.array([-0.0000498810, 0.0, 0.0001117217])
    np.testing.assert_allclose(c, v_ref, rtol=1e-04)


def test_windrose():
    sp = np.array([3.5, 1, 1.1, 2.1, 0.1, 0.0, 2.4, 1.9, 1.7, 3.9, 3.1, 2.1, np.nan, np.nan])
    d = np.array([1.0, 29, 31, 93.0, 121, 171, 189, 245, 240.11, 311, 359.1, np.nan, 11, np.nan])
    sp_bins = [0, 1, 2, 3, 4]

    # count
    v_ref = np.array(
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
        ]
    )
    dir_bin_ref = np.array(
        [
            -30.0000000000,
            30.0000000000,
            90.0000000000,
            150.0000000000,
            210.0000000000,
            270.0000000000,
            330.0000000000,
        ]
    )

    r = wind.windrose(sp, d, sectors=6, speed_bins=sp_bins, percent=False)
    np.testing.assert_allclose(r[0], v_ref, rtol=1e-04)
    np.testing.assert_allclose(r[1], dir_bin_ref, rtol=1e-04)

    # percent
    v_ref = v_ref * 100 / 11.0
    r = wind.windrose(sp, d, sectors=6, speed_bins=sp_bins, percent=True)
    np.testing.assert_allclose(r[0], v_ref, rtol=1e-04)
    np.testing.assert_allclose(r[1], dir_bin_ref, rtol=1e-04)

    # numbers
    sp = 3.4
    d = 90.01
    sp_bins = [0, 5]
    v_ref = np.array([[0, 0, 1, 0, 0, 0]])

    r = wind.windrose(sp, d, sectors=6, speed_bins=sp_bins, percent=False)
    np.testing.assert_allclose(r[0], v_ref, rtol=1e-04)
    np.testing.assert_allclose(r[1], dir_bin_ref, rtol=1e-04)

    # single sector
    sp = 3.4
    d = 90.01
    sp_bins = [0, 5]
    v_ref = np.array([[1]])
    dir_bin_ref = np.array([-180.0000000000, 180.0000000000])

    r = wind.windrose(sp, d, sectors=1, speed_bins=sp_bins, percent=False)
    np.testing.assert_allclose(r[0], v_ref, rtol=1e-04)
    np.testing.assert_allclose(r[1], dir_bin_ref, rtol=1e-04)

    # invalid arguments
    sp = 3.4
    d = 90.01

    sp_bins = [0, 1]
    try:
        r = wind.windrose(sp, d, sectors=0, speed_bins=sp_bins, percent=False)
        assert False
    except ValueError:
        pass

    sp_bins = [0]
    try:
        r = wind.windrose(sp, d, sectors=6, speed_bins=sp_bins, percent=False)
        assert False
    except ValueError:
        pass

    try:
        r = wind.windrose(sp, d, sectors=6, percent=False)
        assert False
    except ValueError:
        pass
