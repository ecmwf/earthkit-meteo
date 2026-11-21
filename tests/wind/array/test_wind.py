# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from earthkit.utils.array.testing import NAMESPACE_DEVICES

import earthkit.meteo.wind.array as wind

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
    "omega,t,p,v_ref",
    [([1.2, 21.3], [285.6, 261.1], [1000, 850], [-0.1003208031, -1.9152219066])],
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


# histogram2d is not available in torch, so we skip these tests for now
HISTOGRAM2D_NS = [
    (ns, dev)
    for ns, dev in NAMESPACE_DEVICES
    if not (ns._earthkit_array_namespace_name == "torch" and "cuda" in dev.type)
]


@pytest.mark.parametrize("xp, device", HISTOGRAM2D_NS)
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
            [1.0, 1.0],
            [11.3, 348.6],
            16,
            [0, 10],
            False,
            None,
            [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
            [-11.25 + 22.5 * i for i in range(17)],
        ),
        (
            1.0,
            22.7,
            8,
            [0, 10],
            False,
            None,
            [[0, 1, 0, 0, 0, 0, 0, 0]],
            [-22.5 + 45.0 * i for i in range(9)],
        ),
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
            # 7 sectors: sectors_start is -360/14, which is not exactly representable. The sample
            # at the wrap-around (sectors_start + 360) belongs to the first sector.
            [1.0, 1.0, 1.0, 1.0],
            [334.2857142857143, 11.3, 150, 5],
            7,
            [0, 10],
            False,
            None,
            [[3, 0, 0, 1, 0, 0, 0]],
            np.linspace(-360.0 / 14, 360.0 - 360.0 / 14, 8),
        ),
    ],
)
def test_windrose_1(sp, d, sectors, sp_bins, percent, sectors_start, v_ref, dir_bin_ref, xp, device):
    sp, d, sp_bins, v_ref, dir_bin_ref = (xp.asarray(x, device=device) for x in [sp, d, sp_bins, v_ref, dir_bin_ref])

    dir_bin_ref = xp.astype(dir_bin_ref, sp.dtype)

    r = wind.windrose(sp, d, sectors=sectors, speed_bins=sp_bins, percent=percent, sectors_start=sectors_start)

    dir_bin_ref = xp.astype(dir_bin_ref, r[0].dtype)
    v_ref = xp.astype(v_ref, r[1].dtype)

    assert xp.allclose(r[0], v_ref, equal_nan=True, rtol=1e-04)
    assert xp.allclose(r[1], dir_bin_ref, equal_nan=True, rtol=1e-04)


def _reduce_direction(value):
    """Reduce a direction into [0, 360)."""
    return float(np.float64(value) % 360.0)


def _edge_samples(values, dtype):
    """``values`` together with their nearest representable neighbours in ``dtype``. The
    neighbours must be computed in the dtype they are used in: computed in float64 and then cast
    to float32 they collapse onto the value itself and are not tested any more.
    """
    samples = []
    for v in values:
        v = np.array(v, dtype=dtype)
        samples += [v.item(), np.nextafter(v, -np.inf).item(), np.nextafter(v, np.inf).item()]
    return samples


def _sector_reference(value, start, sectors):
    """Brute-force reference for the sector a direction belongs to. A direction at the
    wrap-around (sectors_start + 360) belongs to the first sector.
    """
    step = 360.0 / sectors
    return min(int(((float(value) - start) % 360.0) // step), sectors - 1)


# Sector counts whose step (360 / sectors) is not exactly representable in floating point are
# the interesting ones: the sample at sectors_start + 360 used to be wrapped slightly below the
# first direction bin edge and silently dropped by the histogram. This is not an odd/even
# property (it also affects the even count 22), so both are listed here.
@pytest.mark.parametrize("xp, device", HISTOGRAM2D_NS)
@pytest.mark.parametrize("sectors", [3, 5, 7, 8, 9, 11, 13, 22])
@pytest.mark.parametrize("sectors_start", [None, 0, 12.5, -50, -180, 179.99])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_windrose_wrap_around(xp, device, sectors, sectors_start, dtype):
    np_dtype = np.dtype(dtype)
    dtype = getattr(xp, dtype)
    step = 360.0 / sectors
    start = -step / 2 if sectors_start is None else float(sectors_start)

    edges = [_reduce_direction(start + k * step) for k in range(sectors + 1)]
    mids = [_reduce_direction(start + (k + 0.5) * step) for k in range(sectors)]
    # 0, 360, every sector edge with its nearest representable neighbours, and the middle of
    # each sector
    near_edges = _edge_samples(edges, np_dtype)
    directions = [0.0, 360.0] + near_edges + mids
    speeds = [5.0] * len(directions)

    d = xp.asarray(directions, dtype=dtype, device=device)
    sp = xp.asarray(speeds, dtype=dtype, device=device)
    kwargs = dict(sectors=sectors, speed_bins=[0, 10], sectors_start=sectors_start)

    res, dir_bins = wind.windrose(sp, d, percent=False, **kwargs)
    # no sample may be lost, whatever the sector count
    assert int(xp.sum(res)) == len(directions)

    pct, _ = wind.windrose(sp, d, percent=True, **kwargs)
    assert float(xp.sum(pct)) == pytest.approx(100.0, rel=1e-4)

    # the direction bins are unchanged
    bins_ref = np.linspace(start, start + 360.0, sectors + 1).tolist()
    assert xp.allclose(dir_bins, xp.asarray(bins_ref, dtype=dir_bins.dtype, device=device), rtol=1e-4)

    # the samples in the middle of a sector are far from the edges: they must land in the
    # sector given by an independent reference
    speeds_ref = xp.asarray([5.0] * len(mids), dtype=dtype, device=device)
    res_mid, _ = wind.windrose(speeds_ref, xp.asarray(mids, dtype=dtype, device=device), percent=False, **kwargs)
    expected = [[0.0] * sectors]
    for v in mids:
        expected[0][_sector_reference(v, start, sectors)] += 1
    assert xp.allclose(res_mid, xp.asarray(expected, dtype=res_mid.dtype, device=device), rtol=1e-4)


@pytest.mark.parametrize("xp, device", HISTOGRAM2D_NS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_windrose_wrap_around_seven_sectors(xp, device, dtype):
    # 7 sectors: sectors_start is -360/14, which is not exactly representable, and
    # 334.2857142857143 is sectors_start + 360. It belongs to the first sector, like North.
    dtype = getattr(xp, dtype)
    n_samples = 3
    sp = xp.asarray([5.0] * n_samples, dtype=dtype, device=device)
    d = xp.asarray([334.2857142857143, 100.0, 200.0], dtype=dtype, device=device)

    expected = [0.0] * 7
    for k in (0, 2, 4):
        expected[k] = 1.0

    res, _ = wind.windrose(sp, d, sectors=7, speed_bins=[0, 10], percent=False)
    # no sample may be lost at the wrap-around
    assert int(xp.sum(res)) == n_samples
    assert xp.allclose(res, xp.asarray([expected], dtype=res.dtype, device=device))

    # percentages are relative to all the samples, not only to those that were binned
    pct, _ = wind.windrose(sp, d, sectors=7, speed_bins=[0, 10], percent=True)
    pct_expected = [[v * 100.0 / n_samples for v in expected]]
    assert xp.allclose(pct, xp.asarray(pct_expected, dtype=pct.dtype, device=device))


@pytest.mark.parametrize("xp, device", HISTOGRAM2D_NS)
def test_windrose_wrap_around_mixed_dtype_reproducer(xp, device):
    # The direction bins are built in the dtype of the speed, so a direction provided with a
    # wider dtype must be converted to it before it is wrapped. Otherwise the sample at the
    # wrap-around (sectors_start + 360, e.g. 334.2857142857143 for 7 sectors) stays slightly
    # above the last edge and is silently dropped, or counted in the last sector.
    sp = xp.asarray([5.0, 5.0, 5.0], dtype=xp.float32, device=device)
    d = xp.asarray([334.2857142857143, 0.0, 360.0], dtype=xp.float64, device=device)

    expected = {7: [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 11: [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

    for sectors, ref in expected.items():
        start = -(360.0 / sectors) / 2
        # 0 and 360 are the same bearing, and so is sectors_start + 360: all of them belong to
        # the first sector
        dirs = xp.asarray([start + 360.0, 0.0, 360.0], dtype=xp.float64, device=device)

        res, dir_bins = wind.windrose(sp, dirs, sectors=sectors, speed_bins=[0, 10], percent=False)
        assert dir_bins.dtype == sp.dtype
        assert int(xp.sum(res)) == 3
        assert xp.allclose(res, xp.asarray([ref], dtype=res.dtype, device=device))

        pct, _ = wind.windrose(sp, dirs, sectors=sectors, speed_bins=[0, 10], percent=True)
        assert float(xp.sum(pct)) == pytest.approx(100.0, rel=1e-4)

    # the direction of the original report
    res, _ = wind.windrose(sp, d, sectors=7, speed_bins=[0, 10], percent=False)
    assert int(xp.sum(res)) == 3
    res, _ = wind.windrose(sp, d, sectors=11, speed_bins=[0, 10], percent=False)
    assert int(xp.sum(res)) == 3


# Sector counts whose step (360 / sectors) is not exactly representable are the interesting ones
# also when speed and direction have different dtypes: the direction bins are built in the speed
# dtype, so the directions must be converted to it before they are wrapped into
# [sectors_start, sectors_start + 360).
@pytest.mark.parametrize("xp, device", HISTOGRAM2D_NS)
@pytest.mark.parametrize("sectors", [3, 5, 7, 8, 11, 13])
@pytest.mark.parametrize("sectors_start", [None, 0, -90, 179.99])
@pytest.mark.parametrize("sp_dtype,d_dtype", [("float32", "float64"), ("float64", "float32")])
def test_windrose_wrap_around_mixed_dtypes(xp, device, sectors, sectors_start, sp_dtype, d_dtype):
    sp_np_dtype, d_np_dtype = np.dtype(sp_dtype), np.dtype(d_dtype)
    sp_dtype, d_dtype = getattr(xp, sp_dtype), getattr(xp, d_dtype)

    step = 360.0 / sectors
    start = -step / 2 if sectors_start is None else float(sectors_start)

    # 0, 360 and every sector edge with its nearest representable neighbours. The edges are the
    # ones windrose() builds, i.e. in the speed dtype, while the neighbours belong to the dtype
    # the directions are stored in
    edges = np.linspace(start, start + 360.0, sectors + 1, dtype=sp_np_dtype).astype(d_np_dtype)
    directions = _edge_samples([0.0, 360.0, *edges], d_np_dtype)
    speeds = [5.0] * len(directions)

    sp = xp.asarray(speeds, dtype=sp_dtype, device=device)
    d = xp.asarray(directions, dtype=d_dtype, device=device)
    kwargs = dict(sectors=sectors, speed_bins=[0, 10], sectors_start=sectors_start)

    res, dir_bins = wind.windrose(sp, d, percent=False, **kwargs)
    # no sample may be lost when the dtypes of speed and direction differ
    assert int(xp.sum(res)) == len(directions)

    pct, _ = wind.windrose(sp, d, percent=True, **kwargs)
    assert float(xp.sum(pct)) == pytest.approx(100.0, rel=1e-4)

    # the direction bins are unchanged: same values and still the dtype of the speed
    assert dir_bins.dtype == sp_dtype
    bins_ref = np.linspace(start, start + 360.0, sectors + 1, dtype=sp_np_dtype)
    assert xp.all(dir_bins == xp.asarray(bins_ref, dtype=sp_dtype, device=device))


# The direction bins are built in the dtype of the speed, but they must not be used to convert the
# directions unless that dtype is floating point: with an integer speed the conversion would
# truncate the directions and turn a NaN direction into a value that the histogram counts. The
# integer sector counts below (4 sectors: -45, 45, 135, 225 and 315) have exactly representable
# edges, so the bins themselves are not truncated.
@pytest.mark.parametrize("xp, device", HISTOGRAM2D_NS)
@pytest.mark.parametrize("sp_dtype", ["int32", "int64"])
@pytest.mark.parametrize("d_dtype", ["float32", "float64"])
def test_windrose_integer_speed_nan_direction(xp, device, sp_dtype, d_dtype):
    sp = xp.asarray([5, 5], dtype=getattr(xp, sp_dtype), device=device)
    d = xp.asarray([np.nan, 90.0], dtype=getattr(xp, d_dtype), device=device)

    res, dir_bins = wind.windrose(sp, d, sectors=4, speed_bins=[0, 10], percent=False)
    # the sample with the NaN direction is ignored
    assert int(xp.sum(res)) == 1
    assert xp.allclose(res, xp.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=res.dtype, device=device))

    # the direction bins are unchanged: same values and still the dtype of the speed
    assert dir_bins.dtype == sp.dtype
    assert xp.all(dir_bins == xp.asarray([-45, 45, 135, 225, 315], dtype=sp.dtype, device=device))


@pytest.mark.parametrize("xp, device", HISTOGRAM2D_NS)
@pytest.mark.parametrize("sp_dtype", ["int32", "int64"])
@pytest.mark.parametrize("d_dtype", ["float32", "float64"])
def test_windrose_integer_speed_wrap_around(xp, device, sp_dtype, d_dtype):
    sp = xp.asarray([5, 5, 5, 5], dtype=getattr(xp, sp_dtype), device=device)
    # -45.5, just below the first edge (-45), is wrapped into the last sector instead of being
    # truncated onto the first one, while the wrap-around (sectors_start + 360 = 315) belongs to
    # the first sector, like 0 and 360
    d = xp.asarray([-45.5, 315.0, 0.0, 360.0], dtype=getattr(xp, d_dtype), device=device)

    res, dir_bins = wind.windrose(sp, d, sectors=4, speed_bins=[0, 10], percent=False)
    assert int(xp.sum(res)) == 4
    assert xp.allclose(res, xp.asarray([[3.0, 0.0, 0.0, 1.0]], dtype=res.dtype, device=device))

    pct, _ = wind.windrose(sp, d, sectors=4, speed_bins=[0, 10], percent=True)
    assert float(xp.sum(pct)) == pytest.approx(100.0, rel=1e-4)

    # the direction bins are unchanged: same values and still the dtype of the speed
    assert dir_bins.dtype == sp.dtype
    assert xp.all(dir_bins == xp.asarray([-45, 45, 135, 225, 315], dtype=sp.dtype, device=device))


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "sp,d,sectors,sp_bins",
    [(3.4, 90.01, 0, [0, 1]), (3.4, 90.01, 6, [0]), (3.4, 90.01, 6, None)],
)
def test_windrose_invalid(sp, d, sectors, sp_bins, xp, device):
    if sp_bins is not None:
        sp_bins = xp.asarray(sp_bins)
    sp, d = xp.asarray(sp, device=device), xp.asarray(d, device=device)

    with pytest.raises(ValueError):
        wind.windrose(sp, d, sectors=sectors, speed_bins=sp_bins, percent=False)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("sectors_start", [-180.5, 180, 360])
def test_windrose_invalid_sectors_start(sectors_start, xp, device):
    sp, d = xp.asarray(3.4, device=device), xp.asarray(90.01, device=device)

    with pytest.raises(ValueError):
        wind.windrose(sp, d, sectors=8, speed_bins=xp.asarray([0, 5]), percent=False, sectors_start=sectors_start)
