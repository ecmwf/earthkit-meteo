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
from earthkit.utils.array.convert import convert
from earthkit.utils.array.testing import NAMESPACE_DEVICES

import earthkit.meteo.vertical as vertical_high_level
import earthkit.meteo.vertical.array as vertical_array
from earthkit.meteo import constants
from earthkit.meteo.utils.testing import Tolerance


@pytest.fixture
def vertical():
    return vertical_array


# ----------------------------------------------------------------------------
# Test data (numpy float64; backend arrays are created from the .tolist()
# representations so that the default dtype of each backend is used).
# ----------------------------------------------------------------------------

G = constants.g

# pressure levels (Pa), data (3, 2) and the trapezoidal weights
# w0 = (20000 - 10000) / 2, w1 = (40000 - 10000) / 2, w2 = (40000 - 20000) / 2
P = np.array([10000.0, 20000.0, 40000.0])
W_P = np.array([5000.0, 15000.0, 10000.0])
F = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
EXP_P = F.T @ W_P / G  # (65000, 130000) / g

# pressure interfaces (N + 1 values) and layer pressure differences
P_INTF = np.array([5000.0, 15000.0, 30000.0, 70000.0])
DP_INTF = np.diff(P_INTF)  # (10000, 15000, 40000)
EXP_P_INTF = (F * DP_INTF[:, None]).sum(axis=0) / G  # (160000, 320000) / g

# heights (m), trapezoidal weights, interfaces, densities
Z = np.array([0.0, 1000.0, 2000.0])
W_Z = np.array([500.0, 1000.0, 500.0])
EXP_Z = F.T @ W_Z  # (4000, 8000)
Z_INTF = np.array([100.0, 600.0, 1500.0, 3000.0])
DZ_INTF = np.diff(Z_INTF)  # (500, 900, 1500)
EXP_Z_INTF = (F * DZ_INTF[:, None]).sum(axis=0)  # (6800, 13600)
RHO = np.array([1.2, 0.9, 0.6])  # kg m-3 on the data levels
EXP_RHO = (F * (W_Z * RHO)[:, None]).sum(axis=0)  # (3300, 6600)

# synthetic hybrid (IFS model) level definition: 5 full levels, 6 half levels
NLEV = 5
A_HYBRID = np.linspace(0.0, 1000.0, NLEV + 1)  # Pa
B_HYBRID = np.linspace(0.0, 1.0, NLEV + 1)
SP = np.array([100000.0, 98000.0])  # surface pressure (Pa) at two grid points
F_HYBRID = np.arange(1.0, NLEV + 1)[:, None] * np.ones((1, len(SP)))  # (NLEV, 2)

# tolerance of the backends (the tests are exact in float64; float32 needs slack)
TOL = Tolerance({64: (1e-8, 1e-6), 32: (1e-3, 1e-4)})


def _to_backend(xp, device, *objs):
    return tuple(xp.asarray(o, device=device) for o in objs)


def _lists(*arrays):
    return [a.tolist() if isinstance(a, np.ndarray) else a for a in arrays]


def _as_numpy(a):
    # results are compared in numpy on the CPU: the earthkit-utils isclose
    # helper fails when a torch MPS result is compared to an array living on
    # another device, and the torch array-api wrapper rejects negative strides
    return np.asarray(convert(a, array_namespace="numpy", device="cpu"), dtype="float64")


def _check(res, expected, dtype):
    atol, rtol = TOL.get(dtype=dtype)
    np.testing.assert_allclose(_as_numpy(res), np.asarray(expected, dtype="float64"), atol=atol, rtol=rtol)


def _hybrid_interfaces(idx):
    # interface pressures p_{k+1/2} = A_{k+1/2} + B_{k+1/2} * sp computed
    # independently from the A and B coefficients and the surface pressure,
    # with the half-level numbers in ``idx`` (vertical axis first)
    return A_HYBRID[idx][:, None] + B_HYBRID[idx][:, None] * SP[None, :]


def _interface_expected(data_np, p):
    # sum_k f_k * |p_{k+1} - p_k| / g with p computed by _hybrid_interfaces
    return (data_np * np.abs(np.diff(p, axis=0))).sum(axis=0) / G


# ----------------------------------------------------------------------------
# pressure: coordinates on the data levels (trapezoidal rule)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_pressure_trapezoid_manual(vertical, xp, device):
    # hand-computed example: sum(f_i * w_i) / g with w = [5000, 15000, 10000] Pa
    # column 0: (1 * 5000 + 2 * 15000 + 3 * 10000) / g = 65000 / g
    data, p = _to_backend(xp, device, *_lists(F, P))
    res = vertical.vertical_integral_pressure(data, p)
    assert res.shape == EXP_P.shape
    _check(res, EXP_P, data.dtype)

    # single column: 1D data gives a scalar result
    res_1d = vertical.vertical_integral_pressure(data[:, 0], p)
    assert _as_numpy(res_1d).shape == ()
    _check(res_1d, EXP_P[0], data.dtype)

    # a column given as (N, 1) returns shape (1,)
    res_col = vertical.vertical_integral_pressure(data[:, :1], p)
    assert _as_numpy(res_col).shape == (1,)
    _check(res_col, EXP_P[:1], data.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_pressure_ascending_equals_descending(vertical, xp, device):
    # the absolute differences make the direction of the coordinate irrelevant
    data_up, p_up = _to_backend(xp, device, *_lists(F, P))
    data_down, p_down = _to_backend(xp, device, *_lists(F[::-1], P[::-1]))

    res_up = vertical.vertical_integral_pressure(data_up, p_up)
    res_down = vertical.vertical_integral_pressure(data_down, p_down)
    _check(res_up, EXP_P, data_up.dtype)
    _check(res_down, EXP_P, data_up.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_pressure_constant_field(vertical, xp, device):
    # the trapezoidal rule is exact for a constant field: I = c * (p_bot - p_top) / g
    data_np = np.full((len(P), 2), 3.0)
    expected = np.full(2, 3.0 * (P[-1] - P[0]) / G)

    data, p = _to_backend(xp, device, *_lists(data_np, P))
    res = vertical.vertical_integral_pressure(data, p)
    _check(res, expected, data.dtype)

    # the descending order gives the same result
    data_d, p_d = _to_backend(xp, device, *_lists(data_np[::-1], P[::-1]))
    res = vertical.vertical_integral_pressure(data_d, p_d)
    _check(res, expected, data.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_pressure_f_equals_p(vertical, xp, device):
    # f = p: the trapezoidal rule integrates p exactly, giving
    # I = (p_bot^2 - p_top^2) / (2 g)
    data_np = P[:, None] * np.ones((1, 2))
    expected = np.full(2, (P[-1] ** 2 - P[0] ** 2) / (2 * G))

    data, p = _to_backend(xp, device, *_lists(data_np, P))
    res = vertical.vertical_integral_pressure(data, p)
    _check(res, expected, data.dtype)

    # the descending order gives the same (positive) result
    data_d, p_d = _to_backend(xp, device, *_lists(data_np[::-1], P[::-1]))
    res = vertical.vertical_integral_pressure(data_d, p_d)
    _check(res, expected, data.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_pressure_data_shaped(vertical, xp, device):
    # data-shaped coordinate with a different monotonic order in each column
    p_full = np.tile(P, (2, 1)).T.copy()  # (3, 2)
    p_full[:, 1] = [40000.0, 20000.0, 10000.0]  # descending column
    # column 1 weights: [10000, 15000, 5000] -> (2 * 10000 + 4 * 15000 + 6 * 5000) / g
    expected = np.array([65000.0, 110000.0]) / G

    data, coord = _to_backend(xp, device, *_lists(F, p_full))
    res = vertical.vertical_integral_pressure(data, coord)
    _check(res, expected, data.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_pressure_coord_broadcast(vertical, xp, device):
    # singleton non-vertical dimensions of "p" are broadcast over the data
    data, p = _to_backend(xp, device, *_lists(F, P))

    p_col = xp.asarray(P.reshape(3, 1).tolist(), device=device)  # (N, 1)
    res = vertical.vertical_integral_pressure(data, p_col)
    _check(res, EXP_P, data.dtype)

    # 1D coordinate along a non-default vertical_dim
    d23 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # vertical axis at 1
    p_row = P.reshape(1, 3)  # (1, N) broadcast over the rows
    expected = (d23 * W_P).sum(axis=1) / G
    d23_b, p_row_b = _to_backend(xp, device, *_lists(d23, p_row))
    res = vertical.vertical_integral_pressure(d23_b, p_row_b, vertical_dim=1)
    _check(res, expected, d23_b.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_pressure_data_shaped_nondefault_dim(vertical, xp, device):
    # (1, N, 1) coordinate with the vertical axis at position 1
    p4 = np.arange(1.0, 5.0) * 10000.0
    data_np = np.ones((3, 4, 5))
    coord = p4.reshape(1, 4, 1)
    expected = np.full((3, 5), (p4[-1] - p4[0]) / G)  # constant field: span / g

    data, coord_b = _to_backend(xp, device, *_lists(data_np, coord))
    res = vertical.vertical_integral_pressure(data, coord_b, vertical_dim=1)
    assert res.shape == expected.shape
    _check(res, expected, data.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("vertical_dim", [0, 1, -1])
def test_pressure_vertical_dim(vertical, xp, device, vertical_dim):
    # the same result is obtained regardless of the position of the vertical axis
    rng = np.random.default_rng(seed=7)
    data_0 = rng.random((len(P), 2, 3)) * 10.0
    shift = {0: 0, 1: 1, -1: 2}[vertical_dim]
    data_np = np.moveaxis(data_0, 0, shift)  # vertical axis moved to `vertical_dim`
    expected = np.tensordot(data_0, W_P, axes=(0, 0)) / G  # (2, 3)

    data, p = _to_backend(xp, device, *_lists(data_np, P))
    res = vertical.vertical_integral_pressure(data, p, vertical_dim=vertical_dim)
    assert res.shape == expected.shape
    _check(res, expected, data.dtype)


# ----------------------------------------------------------------------------
# pressure: coordinates on the layer interfaces (N + 1 values)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_pressure_interfaces_manual(vertical, xp, device):
    # hand-computed example: w_k = |p_{k+1} - p_k| = [10000, 15000, 40000] Pa
    # column 0: (1 * 10000 + 2 * 15000 + 3 * 40000) / g = 160000 / g
    data, p = _to_backend(xp, device, *_lists(F, P_INTF))
    res = vertical.vertical_integral_pressure(data, p)
    assert res.shape == EXP_P_INTF.shape
    _check(res, EXP_P_INTF, data.dtype)

    # constant field: I = c * (p_last_interface - p_first_interface) / g
    data_c = np.full((3, 2), 2.0)
    expected_c = np.full(2, 2.0 * (P_INTF[-1] - P_INTF[0]) / G)
    data_c_b, p_int = _to_backend(xp, device, *_lists(data_c, P_INTF))
    res = vertical.vertical_integral_pressure(data_c_b, p_int)
    _check(res, expected_c, data_c_b.dtype)

    # the descending order gives the same result
    data_d, p_d = _to_backend(xp, device, *_lists(F[::-1], P_INTF[::-1]))
    res = vertical.vertical_integral_pressure(data_d, p_d)
    _check(res, EXP_P_INTF, data.dtype)

    # a single level with two interfaces (N = 1) is exact for a layer-constant field
    d1 = np.array([[4.0, 8.0]])
    x1 = np.array([20000.0, 30000.0])
    exp1 = np.array([40000.0, 80000.0]) / G
    d1_b, x1_b = _to_backend(xp, device, *_lists(d1, x1))
    res = vertical.vertical_integral_pressure(d1_b, x1_b)
    _check(res, exp1, d1_b.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_pressure_f_equals_p_interfaces(vertical, xp, device):
    # f sampled at the layer midpoints of the interfaces x_k:
    # sum_k (x_k + x_{k+1}) / 2 * |x_{k+1} - x_k| telescopes to
    # (x_N^2 - x_0^2) / 2, so I = (x_N^2 - x_0^2) / (2 g)
    data_np = (0.5 * (P_INTF[:-1] + P_INTF[1:]))[:, None] * np.ones((1, 2))
    expected = np.full(2, (P_INTF[-1] ** 2 - P_INTF[0] ** 2) / (2 * G))

    data, p = _to_backend(xp, device, *_lists(data_np, P_INTF))
    res = vertical.vertical_integral_pressure(data, p)
    _check(res, expected, data.dtype)

    # the descending order gives the same (positive) result
    data_d, p_d = _to_backend(xp, device, *_lists(data_np[::-1], P_INTF[::-1]))
    res = vertical.vertical_integral_pressure(data_d, p_d)
    _check(res, expected, data.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("vertical_dim", [0, 1, -1])
def test_pressure_interfaces_vertical_dim(vertical, xp, device, vertical_dim):
    # N + 1 = 4 interface values with the vertical axis at any position
    rng = np.random.default_rng(seed=3)
    data_0 = rng.random((3, 2, 4)) * 10.0
    expected = (data_0 * DP_INTF[:, None, None]).sum(axis=0) / G  # (2, 4)

    shift = {0: 0, 1: 1, -1: 2}[vertical_dim]
    data_np = np.moveaxis(data_0, 0, shift)
    data, p = _to_backend(xp, device, *_lists(data_np, P_INTF))
    res = vertical.vertical_integral_pressure(data, p, vertical_dim=vertical_dim)
    assert res.shape == expected.shape
    _check(res, expected, data.dtype)

    # data-shaped interface coordinate with the vertical values at
    # `vertical_dim` and singleton non-vertical dimensions
    coord_shape = [1, 1, 1]
    coord_shape[shift] = len(P_INTF)
    coord_b = xp.asarray(P_INTF.reshape(coord_shape).tolist(), device=device)
    res = vertical.vertical_integral_pressure(data, coord_b, vertical_dim=vertical_dim)
    _check(res, expected, data.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_pressure_interfaces_hybrid_all_levels(vertical, xp, device):
    # all levels: the interfaces come from
    # pressure_on_hybrid_levels(sp, A, B, output="half")
    p_half = vertical.pressure_on_hybrid_levels(SP, A_HYBRID, B_HYBRID, output="half")
    assert p_half.shape == (NLEV + 1, len(SP))

    interfaces = _hybrid_interfaces(np.arange(NLEV + 1))
    np.testing.assert_allclose(np.asarray(p_half), interfaces)
    expected = _interface_expected(F_HYBRID, interfaces)

    data, p = _to_backend(xp, device, *_lists(F_HYBRID, np.asarray(p_half)))
    res = vertical.vertical_integral_pressure(data, p)
    assert res.shape == expected.shape
    _check(res, expected, data.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "levels,level_idx,half_idx",
    [
        ([2, 3, 4], [1, 2, 3], [1, 2, 3, 4]),  # ascending subset -> interfaces 1..4
        ([4, 3, 2], [3, 2, 1], [4, 3, 2, 1]),  # descending subset -> interfaces 4..1
    ],
)
def test_pressure_interfaces_hybrid_subset(vertical, xp, device, levels, level_idx, half_idx):
    # a subset of the full levels: the enveloping interfaces come from
    # pressure_on_hybrid_levels(sp, A, B, levels=levels, output=("full", "half"))
    _, p_half = vertical.pressure_on_hybrid_levels(SP, A_HYBRID, B_HYBRID, levels=levels, output=("full", "half"))
    assert p_half.shape == (len(level_idx) + 1, len(SP))

    interfaces = _hybrid_interfaces(np.asarray(half_idx))
    np.testing.assert_allclose(np.asarray(p_half), interfaces)

    data_np = F_HYBRID[level_idx]
    expected = _interface_expected(data_np, interfaces)

    data, p = _to_backend(xp, device, *_lists(data_np, np.asarray(p_half)))
    res = vertical.vertical_integral_pressure(data, p)
    assert res.shape == expected.shape
    _check(res, expected, data.dtype)


# ----------------------------------------------------------------------------
# height
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_height_trapezoid_manual(vertical, xp, device):
    # hand-computed example: w = [500, 1000, 500] m, no 1/g
    # column 0: (1 * 500 + 2 * 1000 + 3 * 500) = 4000
    data, h = _to_backend(xp, device, *_lists(F, Z))
    res = vertical.vertical_integral_height(data, h)
    assert res.shape == EXP_Z.shape
    _check(res, EXP_Z, data.dtype)

    # constant field: I = c * (h_bot - h_top)
    data_c = np.full((3, 2), 2.0)
    expected_c = np.full(2, 2.0 * (Z[-1] - Z[0]))
    data_c_b, h_b = _to_backend(xp, device, *_lists(data_c, Z))
    res = vertical.vertical_integral_height(data_c_b, h_b)
    _check(res, expected_c, data_c_b.dtype)

    # the descending order gives the same result
    data_d, h_d = _to_backend(xp, device, *_lists(F[::-1], Z[::-1]))
    res = vertical.vertical_integral_height(data_d, h_d)
    _check(res, EXP_Z, data.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_height_interfaces(vertical, xp, device):
    # N + 1 interface heights: w_k = |h_{k+1} - h_k| = [500, 900, 1500] m
    # column 0: (1 * 500 + 2 * 900 + 3 * 1500) = 6800
    data, h = _to_backend(xp, device, *_lists(F, Z_INTF))
    res = vertical.vertical_integral_height(data, h)
    _check(res, EXP_Z_INTF, data.dtype)

    # constant field: I = c * (h_last_interface - h_first_interface)
    data_c = np.full((3, 2), 2.0)
    expected_c = np.full(2, 2.0 * (Z_INTF[-1] - Z_INTF[0]))
    data_c_b, h_b = _to_backend(xp, device, *_lists(data_c, Z_INTF))
    res = vertical.vertical_integral_height(data_c_b, h_b)
    _check(res, expected_c, data_c_b.dtype)

    # the descending order gives the same result
    data_d, h_d = _to_backend(xp, device, *_lists(F[::-1], Z_INTF[::-1]))
    res = vertical.vertical_integral_height(data_d, h_d)
    _check(res, EXP_Z_INTF, data.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_height_rho_1d(vertical, xp, device):
    # mass-weighted integral: I = sum w_i rho_i f_i (no g)
    # column 0: (500 * 1.2 * 1 + 1000 * 0.9 * 2 + 500 * 0.6 * 3) = 3300
    data, h, rho = _to_backend(xp, device, *_lists(F, Z, RHO))
    res = vertical.vertical_integral_height(data, h, rho=rho)
    _check(res, EXP_RHO, data.dtype)

    # rho is applied to the data, not to the weights: a constant rho and data
    # gives rho * c * (h_bot - h_top)
    data_c = np.full((3, 2), 2.0)
    rho_c = np.full(3, 1.5)
    expected = np.full(2, 1.5 * 2.0 * (Z[-1] - Z[0]))
    data_c_b, rho_c_b, h_b = _to_backend(xp, device, *_lists(data_c, rho_c, Z))
    res = vertical.vertical_integral_height(data_c_b, h_b, rho=rho_c_b)
    _check(res, expected, data_c_b.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_height_rho_data_shaped(vertical, xp, device):
    # data-shaped rho with a different profile in each column
    rho_full = np.tile(RHO, (2, 1)).T.copy()  # (3, 2)
    rho_full[:, 1] = [0.6, 0.9, 1.2]  # increasing downwards
    # column 1: (500 * 0.6 * 2 + 1000 * 0.9 * 4 + 500 * 1.2 * 6) = 7800
    expected = (F * W_Z[:, None] * rho_full).sum(axis=0)

    data, h, rho = _to_backend(xp, device, *_lists(F, Z, rho_full))
    res = vertical.vertical_integral_height(data, h, rho=rho)
    _check(res, expected, data.dtype)

    # singleton non-vertical dimension (N, 1) broadcasts like the 1D rho
    rho_col = xp.asarray(RHO.reshape(3, 1).tolist(), device=device)
    res = vertical.vertical_integral_height(data, h, rho=rho_col)
    _check(res, EXP_RHO, data.dtype)

    # rho on the data levels combined with interface heights:
    # column 0: (1 * 500 * 1.2 + 2 * 900 * 0.9 + 3 * 1500 * 0.6) = 4920
    expected_i = (F * DZ_INTF[:, None] * RHO[:, None]).sum(axis=0)
    data_i, h_i, rho_i = _to_backend(xp, device, *_lists(F, Z_INTF, RHO))
    res = vertical.vertical_integral_height(data_i, h_i, rho=rho_i)
    _check(res, expected_i, data_i.dtype)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("vertical_dim", [0, 1, -1])
def test_height_vertical_dim(vertical, xp, device, vertical_dim):
    rng = np.random.default_rng(seed=5)
    data_0 = rng.random((3, 2, 4)) * 10.0
    expected = np.tensordot(data_0, W_Z, axes=(0, 0))
    expected_rho = np.tensordot(data_0 * RHO[:, None, None], W_Z, axes=(0, 0))

    shift = {0: 0, 1: 1, -1: 2}[vertical_dim]
    data_np = np.moveaxis(data_0, 0, shift)
    data, h, rho = _to_backend(xp, device, *_lists(data_np, Z, RHO))

    res = vertical.vertical_integral_height(data, h, vertical_dim=vertical_dim)
    assert res.shape == expected.shape
    _check(res, expected, data.dtype)

    res = vertical.vertical_integral_height(data, h, rho=rho, vertical_dim=vertical_dim)
    assert res.shape == expected_rho.shape
    _check(res, expected_rho, data.dtype)


# ----------------------------------------------------------------------------
# NaN propagation
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_nan_propagates(vertical, xp, device):
    # NaN in the data propagates to the whole column, other columns are unaffected
    data_np = F.copy()
    data_np[1, 0] = np.nan
    data, p = _to_backend(xp, device, *_lists(data_np, P))
    res = vertical.vertical_integral_pressure(data, p)
    assert np.isnan(_as_numpy(res)[0])
    _check(res[1:], EXP_P[1:], data.dtype)

    # NaN in the level coordinate propagates to the whole result
    p_nan = P.copy()
    p_nan[2] = np.nan
    data, p_nan_b = _to_backend(xp, device, *_lists(F, p_nan))
    res = vertical.vertical_integral_pressure(data, p_nan_b)
    assert bool(xp.all(xp.isnan(res)))

    # NaN in the density propagates to the whole column
    rho_nan = RHO.copy()
    rho_nan[0] = np.nan
    data, h, rho = _to_backend(xp, device, *_lists(F, Z, rho_nan))
    res = vertical.vertical_integral_height(data, h, rho=rho)
    assert bool(xp.all(xp.isnan(res)))


# ----------------------------------------------------------------------------
# output shape
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_output_shape(vertical, xp, device):
    # the vertical axis is removed from the output in every case
    shape = (4, 3, 5)
    data, p, h, p_int, h_int, rho = _to_backend(
        xp,
        device,
        *_lists(
            np.ones(shape),
            np.arange(1.0, 5.0) * 10000.0,  # 4 level values
            np.arange(1.0, 5.0) * 100.0,
            np.arange(5.0) * 10000.0,  # 5 interface values
            np.arange(5.0) * 100.0,
            np.linspace(0.5, 1.5, 4),
        ),
    )

    assert vertical.vertical_integral_pressure(data, p).shape == (3, 5)
    assert vertical.vertical_integral_height(data, h).shape == (3, 5)
    assert vertical.vertical_integral_pressure(data, p_int).shape == (3, 5)
    assert vertical.vertical_integral_height(data, h_int).shape == (3, 5)
    assert vertical.vertical_integral_height(data, h, rho=rho).shape == (3, 5)

    # the vertical axis at a non-default position
    data3 = xp.asarray(np.ones((3, 4, 5)).tolist(), device=device)  # 4 levels at axis 1
    assert vertical.vertical_integral_pressure(data3, p, vertical_dim=1).shape == (3, 5)
    assert vertical.vertical_integral_height(data3, h, rho=rho, vertical_dim=1).shape == (3, 5)
    data4 = xp.asarray(np.ones((3, 5, 4)).tolist(), device=device)  # 4 levels at axis -1
    assert vertical.vertical_integral_pressure(data4, p, vertical_dim=-1).shape == (3, 5)
    assert vertical.vertical_integral_height(data4, h, vertical_dim=-1).shape == (3, 5)


# ----------------------------------------------------------------------------
# error cases
# ----------------------------------------------------------------------------


def _coord_error_cases():
    f = F  # (3, 2): three levels
    return [
        # 1D coordinate: length is neither N nor N + 1
        ("pressure", f, np.array([10000.0, 20000.0]), {}),
        ("pressure", f, np.arange(1000.0, 51000.0, 10000.0), {}),  # 5 values
        ("height", f, np.array([0.0, 1000.0]), {}),
        ("height", f, np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0]), {}),  # 5 values
        # data-shaped coordinate: wrong size along the vertical axis
        ("pressure", f, np.zeros((5, 2)), {}),
        ("height", f, np.zeros((2, 2)), {}),
        # coordinate with the wrong number of dimensions (3D for 2D data)
        ("pressure", f, np.zeros((3, 1, 1)), {}),
        ("height", f, np.zeros((4, 1, 1)), {}),
        # non-vertical dimension neither singleton nor matching the data
        ("pressure", f, np.zeros((3, 3)), {}),
        ("height", f, np.zeros((4, 3)), {}),
        ("pressure", f, np.zeros((2, 3)), {"vertical_dim": 1}),
        # a single level with level coordinates (N < 2 needs interfaces instead)
        ("pressure", np.array([[1.0, 2.0]]), np.array([10000.0]), {}),
        ("height", np.array([[1.0, 2.0]]), np.array([0.0]), {}),
    ]


@pytest.mark.parametrize("func,data,coord,kwargs", _coord_error_cases())
def test_coordinate_errors(vertical, func, data, coord, kwargs):
    with pytest.raises(ValueError):
        getattr(vertical, f"vertical_integral_{func}")(data, coord, **kwargs)


def _rho_error_cases():
    f = F  # (3, 2): three levels
    z = np.array([0.0, 1000.0, 2000.0])
    return [
        # rho must be defined on the N data levels
        (f, z, np.array([1.0, 2.0]), {}),  # too few values
        (f, z, np.array([1.0, 2.0, 3.0, 4.0]), {}),  # N + 1 is not allowed for rho
        (f, z, np.zeros((3, 3)), {}),  # non-vertical dimension neither 1 nor 2
        (f, z, np.zeros((3, 2, 1)), {}),  # wrong number of dimensions
        (f, z, np.zeros((2, 3)), dict(vertical_dim=1)),  # vertical axis at the wrong size
    ]


@pytest.mark.parametrize("data,h,rho,kwargs", _rho_error_cases())
def test_rho_errors(vertical, data, h, rho, kwargs):
    with pytest.raises(ValueError):
        vertical.vertical_integral_height(data, h, rho=rho, **kwargs)


# ----------------------------------------------------------------------------
# high-level wrapper consistency
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("vertical_dim", [0, 1, -1])
def test_high_level_wrappers_match_array(vertical_dim):
    rng = np.random.default_rng(seed=11)
    data_0 = rng.random((len(P), 2, 3)) * 10.0
    shift = {0: 0, 1: 1, -1: 2}[vertical_dim]
    data = np.moveaxis(data_0, 0, shift)

    rho_shape = [1, 1, 1]
    rho_shape[shift] = len(RHO)

    cases = [
        ("vertical_integral_pressure", (data, P)),
        ("vertical_integral_pressure", (data, P[::-1])),
        ("vertical_integral_pressure", (data, P_INTF)),  # N + 1 interfaces
        ("vertical_integral_height", (data, Z)),
        ("vertical_integral_height", (data, Z_INTF, RHO)),  # interfaces + rho
        ("vertical_integral_height", (data, Z, RHO.reshape(rho_shape))),  # broadcastable rho
    ]

    for name, args in cases:
        kwargs = dict(vertical_dim=vertical_dim)
        res_high = getattr(vertical_high_level, name)(*args, **kwargs)
        res_array = getattr(vertical_array, name)(*args, **kwargs)
        assert isinstance(res_high, np.ndarray)
        assert res_high.shape == res_array.shape
        np.testing.assert_allclose(res_high, res_array)
