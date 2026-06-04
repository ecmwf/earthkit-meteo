# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""
Tests for the CAPE/CIN functions.

Input data and reference values are stored in tests/data/cape_cin_input.csv
and tests/data/cape_cin_expected.csv.  Use save_cape_cin_reference() to
regenerate those files when the expected values change.
"""

import os

import numpy as np
import pytest

from earthkit.meteo import thermo
from earthkit.meteo.thermo.array.cape_cin import (
    _lfc_index,
    _vertical_weighted_mean,
    _where_is_param_zero,
)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

CASE_NAMES = ["stable", "elevated_instability", "unstable", "large_cape_small_cin"]
PARCEL_TYPES = ["surface", "mixed", "mu"]


def data_file(name):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", name)


def read_data_file(path):
    return np.genfromtxt(data_file(path), delimiter=",", names=True)


def save_cape_cin_reference(input_data, expected_data):
    """Regenerate the CSV reference files from dicts of 1-D arrays."""
    np.savetxt(
        data_file("cape_cin_input.csv"),
        np.column_stack(list(input_data.values())),
        delimiter=",",
        header=",".join(input_data.keys()),
    )
    np.savetxt(
        data_file("cape_cin_expected.csv"),
        np.column_stack(list(expected_data.values())),
        delimiter=",",
        header=",".join(expected_data.keys()),
    )


class CapeCinData:
    """Load CAPE/CIN test data from the reference CSV files.

    Attributes
    ----------
    p, t, zh, r : dict[str, np.ndarray]
        Per-case 1-D arrays (shape ``(nz,)``), keyed by case name.
        Pressure is in Pa, temperature in K, height in m, mixing ratio in kg/kg.
    p_stacked, t_stacked, zh_stacked, r_stacked : np.ndarray
        All four cases stacked column-wise (shape ``(nz, ncases)``).
    expected_cape, expected_cin : dict[str, np.ndarray]
        Per-parcel-type arrays of expected values (shape ``(ncases,)``).
    """

    def __init__(self):
        input = read_data_file("cape_cin_input.csv")
        expected = read_data_file("cape_cin_expected.csv")

        self.p = {n: input[f"{n}_p"] for n in CASE_NAMES}
        self.t = {n: input[f"{n}_t"] for n in CASE_NAMES}
        self.zh = {n: input[f"{n}_zh"] for n in CASE_NAMES}
        self.r = {n: input[f"{n}_r"] for n in CASE_NAMES}

        self.p_stacked = np.column_stack([self.p[n] for n in CASE_NAMES])
        self.t_stacked = np.column_stack([self.t[n] for n in CASE_NAMES])
        self.zh_stacked = np.column_stack([self.zh[n] for n in CASE_NAMES])
        self.r_stacked = np.column_stack([self.r[n] for n in CASE_NAMES])

        self.expected_cape = {parcel_type: expected[f"{parcel_type}_cape"] for parcel_type in PARCEL_TYPES}
        self.expected_cin = {parcel_type: expected[f"{parcel_type}_cin"] for parcel_type in PARCEL_TYPES}


# ---------------------------------------------------------------------------
# Per-case, per-parcel-type correctness tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parcel_type", PARCEL_TYPES)
@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_cape_cin(case_name, parcel_type):
    data = CapeCinData()
    p = data.p[case_name][:, None]
    t = data.t[case_name][:, None]
    zh = data.zh[case_name][:, None]
    r = data.r[case_name][:, None]

    cape, cin = thermo.cape_cin(p, zh, t, r, parcel_type)

    case_idx = CASE_NAMES.index(case_name)
    np.testing.assert_allclose(cape, data.expected_cape[parcel_type][case_idx], atol=1)
    np.testing.assert_allclose(cin, data.expected_cin[parcel_type][case_idx], atol=1)


# ---------------------------------------------------------------------------
# Multi-profile (stacked) and shape tests
# ---------------------------------------------------------------------------


def test_cape_cin_stacked():
    data = CapeCinData()
    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(data.p_stacked, data.zh_stacked, data.t_stacked, data.r_stacked, parcel_type)
        np.testing.assert_allclose(cape, data.expected_cape[parcel_type], atol=1)
        np.testing.assert_allclose(cin, data.expected_cin[parcel_type], atol=1)


def test_cape_cin_vertical_axis_minus_1():
    data = CapeCinData()
    p = data.p_stacked.T
    t = data.t_stacked.T
    r = data.r_stacked.T
    zh = data.zh_stacked.T

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p, zh, t, r, parcel_type, vertical_axis=-1)
        np.testing.assert_allclose(cape, data.expected_cape[parcel_type], atol=1)
        np.testing.assert_allclose(cin, data.expected_cin[parcel_type], atol=1)


def test_cape_cin_arbitrary_nd_shape():
    nz = 3
    horizontal_shape = (2, 3, 4)
    shape = (nz,) + horizontal_shape

    p = np.broadcast_to(np.array([100000, 90000, 80000])[:, None, None, None], shape)
    t = np.broadcast_to(np.array([290, 280, 270])[:, None, None, None], shape)
    r = np.broadcast_to(np.array([0.01, 0.02, 0.03])[:, None, None, None], shape)
    zh = np.broadcast_to(np.array([100, 200, 300])[:, None, None, None], shape)

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p, zh, t, r, parcel_type)
        assert cape.shape == horizontal_shape
        assert cin.shape == horizontal_shape


def test_cape_cin_lat_lon():
    data = CapeCinData()
    nz = data.p_stacked.shape[0]
    ny, nx = 2, 2
    p = data.p_stacked.reshape(nz, ny, nx)
    t = data.t_stacked.reshape(nz, ny, nx)
    r = data.r_stacked.reshape(nz, ny, nx)
    zh = data.zh_stacked.reshape(nz, ny, nx)

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p, zh, t, r, parcel_type)
        np.testing.assert_allclose(cape, data.expected_cape[parcel_type].reshape(ny, nx), atol=1)
        np.testing.assert_allclose(cin, data.expected_cin[parcel_type].reshape(ny, nx), atol=1)


def test_cape_cin_missing_values():
    data = CapeCinData()
    p = data.p_stacked.copy()
    t = data.t_stacked.copy()
    r = data.r_stacked.copy()
    zh = data.zh_stacked.copy()

    # Introduce NaNs: column 0 via t, column 2 via r
    t[0, 0] = np.nan
    r[3, 2] = np.nan

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p, zh, t, r, parcel_type)
        assert np.isnan(cape[0])
        assert np.isnan(cin[0])
        assert np.isnan(cape[2])
        assert np.isnan(cin[2])


def test_cape_cin_options_forwarded():
    """Regression: options passed to cape_cin() must reach the subclass."""
    data = CapeCinData()
    p = data.p["unstable"][:, None]
    t = data.t["unstable"][:, None]
    r = data.r["unstable"][:, None]
    zh = data.zh["unstable"][:, None]

    # Default layer_depth (5000 Pa) vs a wider mixed layer (15000 Pa) must differ.
    cape_default, _ = thermo.cape_cin(p, zh, t, r, "mixed")
    cape_wide, _ = thermo.cape_cin(p, zh, t, r, "mixed", layer_depth=15000)
    assert not np.isclose(cape_default, cape_wide, atol=1), (
        "layer_depth option was not forwarded to the mixed-layer parcel computation"
    )


def test_cape_cin_invalid_parcel_type():
    """cape_cin() must raise ValueError for an unrecognised parcel_type."""
    data = CapeCinData()
    p = data.p["unstable"][:, None]
    t = data.t["unstable"][:, None]
    r = data.r["unstable"][:, None]
    zh = data.zh["unstable"][:, None]

    with pytest.raises(ValueError, match="parcel_type"):
        thermo.cape_cin(p, zh, t, r, "unknown_parcel")


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


def test_cape_cin_all_nan_profile():
    """An entirely NaN profile must produce NaN outputs without raising.

    Two-column input: column 0 is all-NaN, column 1 is a real stable profile
    that should produce CAPE=0, CIN=0 regardless.
    """
    data = CapeCinData()
    p_real = data.p["stable"][:, None]
    t_real = data.t["stable"][:, None]
    zh_real = data.zh["stable"][:, None]
    r_real = data.r["stable"][:, None]

    p = np.hstack([np.full_like(p_real, np.nan), p_real])
    t = np.hstack([np.full_like(t_real, np.nan), t_real])
    zh = np.hstack([np.full_like(zh_real, np.nan), zh_real])
    r = np.hstack([np.full_like(r_real, np.nan), r_real])

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p, zh, t, r, parcel_type)
        assert np.isnan(cape[0]), f"{parcel_type}: expected NaN for all-NaN column, got {cape[0]}"
        assert np.isnan(cin[0]), f"{parcel_type}: expected NaN for all-NaN column, got {cin[0]}"
        np.testing.assert_allclose(cape[1], 0.0, atol=1)
        np.testing.assert_allclose(cin[1], 0.0, atol=1)


def test_cape_cin_no_lfc():
    """A strongly stable (isothermal) profile must return CAPE=0, CIN=0."""
    data = CapeCinData()
    p = data.p["stable"][:, None]
    zh = data.zh["stable"][:, None]
    r = data.r["stable"][:, None]
    # Isothermal profile: parcel lifted dry-adiabatically is always cooler than env.
    t = np.full_like(p, 260.0)

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p, zh, t, r, parcel_type)
        np.testing.assert_allclose(cape, 0.0, atol=1e-6, err_msg=f"{parcel_type}: expected CAPE=0")
        np.testing.assert_allclose(cin, 0.0, atol=1e-6, err_msg=f"{parcel_type}: expected CIN=0")


def test_cape_cin_unsorted_pressure():
    """A vertically flipped (surface-first) profile must give the same result as sorted."""
    data = CapeCinData()
    p = data.p["unstable"][:, None]
    t = data.t["unstable"][:, None]
    zh = data.zh["unstable"][:, None]
    r = data.r["unstable"][:, None]

    p_flip = np.flip(p, axis=0)
    t_flip = np.flip(t, axis=0)
    zh_flip = np.flip(zh, axis=0)
    r_flip = np.flip(r, axis=0)

    for parcel_type in PARCEL_TYPES:
        cape_sorted, cin_sorted = thermo.cape_cin(p, zh, t, r, parcel_type)
        cape_flip, cin_flip = thermo.cape_cin(p_flip, zh_flip, t_flip, r_flip, parcel_type)
        np.testing.assert_allclose(
            cape_flip, cape_sorted, atol=1, err_msg=f"{parcel_type}: CAPE differs for flipped input"
        )
        np.testing.assert_allclose(
            cin_flip, cin_sorted, atol=1, err_msg=f"{parcel_type}: CIN differs for flipped input"
        )


def test_cape_cin_unsorted_pressure_stacked():
    """Mixed stacked input (one sorted, one unsorted column) must sort both correctly."""
    data = CapeCinData()
    p = data.p["unstable"][:, None]
    t = data.t["unstable"][:, None]
    zh = data.zh["unstable"][:, None]
    r = data.r["unstable"][:, None]

    p_flip = np.flip(p, axis=0)
    t_flip = np.flip(t, axis=0)
    zh_flip = np.flip(zh, axis=0)
    r_flip = np.flip(r, axis=0)

    # col0 = flipped (unsorted), col1 = sorted — both should give the same result
    p2 = np.hstack([p_flip, p])
    t2 = np.hstack([t_flip, t])
    zh2 = np.hstack([zh_flip, zh])
    r2 = np.hstack([r_flip, r])

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p2, zh2, t2, r2, parcel_type)
        np.testing.assert_allclose(
            cape[0],
            cape[1],
            atol=1,
            err_msg=f"{parcel_type}: CAPE mismatch between flipped and sorted columns",
        )
        np.testing.assert_allclose(
            cin[0], cin[1], atol=1, err_msg=f"{parcel_type}: CIN mismatch between flipped and sorted columns"
        )


def test_cape_cin_very_dry():
    """A near-zero mixing ratio profile must return CAPE=0, CIN=0 without raising."""
    data = CapeCinData()
    p = data.p["stable"][:, None]
    zh = data.zh["stable"][:, None]
    t = data.t["stable"][:, None]
    r = np.full_like(p, 1e-9)  # effectively bone-dry

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p, zh, t, r, parcel_type)
        np.testing.assert_allclose(cape, 0.0, atol=1, err_msg=f"{parcel_type}: expected CAPE=0 for dry profile")
        np.testing.assert_allclose(cin, 0.0, atol=1, err_msg=f"{parcel_type}: expected CIN=0 for dry profile")


# ---------------------------------------------------------------------------
# Unit tests for internal helper functions
# ---------------------------------------------------------------------------


class TestWhereIsParamZero:
    """_where_is_param_zero: linear interpolation to find the pressure at which param==0."""

    def test_exact_zero_at_level(self):
        # param crosses zero between index 1 (below) and index 2 (above)
        # p increases with index (ascending pressure)
        p = np.array([[90000.0], [95000.0], [100000.0]])
        param = np.array([[1.0], [0.0], [-1.0]])
        level = np.array([2])  # zero crossing is between level 1 and 2
        result = _where_is_param_zero(level, p, param)
        np.testing.assert_allclose(result, [95000.0], atol=1e-6)

    def test_midpoint_crossing(self):
        # param goes from +2 at level 0 to -2 at level 1 → zero at midpoint pressure
        p = np.array([[80000.0], [90000.0]])
        param = np.array([[2.0], [-2.0]])
        level = np.array([1])
        result = _where_is_param_zero(level, p, param)
        np.testing.assert_allclose(result, [85000.0], atol=1e-6)

    def test_multiple_profiles(self):
        # Two independent profiles
        p = np.array([[80000.0, 80000.0], [90000.0, 90000.0]])
        param = np.array([[2.0, 4.0], [-2.0, -1.0]])
        level = np.array([1, 1])
        result = _where_is_param_zero(level, p, param)
        np.testing.assert_allclose(result, [85000.0, 88000.0], atol=1e-6)

    def test_arbitrary_nd_shape(self):
        # 2×2 horizontal grid, 3 vertical levels
        p = np.broadcast_to(np.array([80000.0, 90000.0, 100000.0])[:, None, None], (3, 2, 2)).copy()
        param = np.broadcast_to(np.array([2.0, 0.0, -2.0])[:, None, None], (3, 2, 2)).copy()
        level = np.full((2, 2), 2)
        result = _where_is_param_zero(level, p, param)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, np.full((2, 2), 90000.0), atol=1e-6)


class TestVerticalWeightedMean:
    """_vertical_weighted_mean: pressure-weighted mean over a layer."""

    def test_uniform_profile(self):
        # Constant param → mean equals that constant regardless of layer
        p = np.array([80000.0, 90000.0, 100000.0])[:, None]
        param = np.full_like(p, 5.0)
        result = _vertical_weighted_mean(p, param, np.array([100000.0]), np.array([80000.0]))
        np.testing.assert_allclose(result, [5.0], rtol=1e-6)

    def test_linear_profile(self):
        # Linearly varying param, uniform dp layers → mean = midpoint value
        p = np.array([80000.0, 90000.0, 100000.0])[:, None]
        param = np.array([1.0, 2.0, 3.0])[:, None]
        result = _vertical_weighted_mean(p, param, np.array([100000.0]), np.array([80000.0]))
        np.testing.assert_allclose(result, [2.0], rtol=1e-6)

    def test_partial_layer(self):
        # Only the bottom layer (90000–100000 Pa) is inside the integration window
        p = np.array([80000.0, 90000.0, 100000.0])[:, None]
        param = np.array([0.0, 10.0, 20.0])[:, None]
        result = _vertical_weighted_mean(p, param, np.array([100000.0]), np.array([90000.0]))
        np.testing.assert_allclose(result, [15.0], rtol=1e-6)

    def test_multiple_profiles(self):
        # Two columns with identical pressures but different params
        p = np.array([[80000.0, 80000.0], [90000.0, 90000.0], [100000.0, 100000.0]])
        param = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        result = _vertical_weighted_mean(p, param, np.array([100000.0, 100000.0]), np.array([80000.0, 80000.0]))
        np.testing.assert_allclose(result, [2.0, 4.0], rtol=1e-6)


class TestLfcIndex:
    """_lfc_index: find the index of the Level of Free Convection."""

    def _make_inputs(self, buoyancy_at_levels, min_depth=0.0):
        """Build z/b/z_lcl arrays from a list of (height, buoyancy) tuples (descending z)."""
        z = np.array([h for h, _ in buoyancy_at_levels], dtype=float)[:, None]
        b = np.array([bv for _, bv in buoyancy_at_levels], dtype=float)[:, None]
        z_lcl = np.array([0.0])  # LCL at ground — no restriction
        return z, b, z_lcl, min_depth

    def test_no_buoyancy_returns_zero(self):
        # All negative buoyancy → no LFC → index 0
        z, b, z_lcl, _ = self._make_inputs([(5000, -1), (3000, -1), (1000, -1)])
        result = _lfc_index(z, b, z_lcl)
        assert result[0] == 0

    def test_single_buoyant_layer(self):
        # Buoyant only at index 0 (top level), no min_depth required
        z = np.array([5000.0, 3000.0, 1000.0])[:, None]
        b = np.array([1.0, -1.0, -1.0])[:, None]
        z_lcl = np.array([0.0])
        result = _lfc_index(z, b, z_lcl, min_depth=0.0)
        # Buoyancy exists → result should be non-zero
        assert result[0] != 0

    def test_deep_buoyant_layer(self):
        # Levels at 5000, 4000, 3000, 2000, 1000 m; buoyant at top four (depth = 3000 m)
        z = np.array([5000.0, 4000.0, 3000.0, 2000.0, 1000.0])[:, None]
        b = np.array([1.0, 1.0, 1.0, 1.0, -1.0])[:, None]
        z_lcl = np.array([0.0])
        # min_depth=2500: layer depth 3000 m qualifies → LFC should exist
        result = _lfc_index(z, b, z_lcl, min_depth=2500.0)
        assert result[0] != 0

    def test_shallow_buoyancy_below_min_depth(self):
        # Buoyant layer depth = 1000 m, min_depth = 1500 m → no LFC
        z = np.array([5000.0, 4000.0, 3000.0, 2000.0])[:, None]
        b = np.array([1.0, -1.0, -1.0, -1.0])[:, None]
        z_lcl = np.array([0.0])
        result = _lfc_index(z, b, z_lcl, min_depth=1500.0)
        assert result[0] == 0

    def test_lcl_mask(self):
        # All levels buoyant, but LCL is above them all → no LFC
        z = np.array([3000.0, 2000.0, 1000.0])[:, None]
        b = np.array([1.0, 1.0, 1.0])[:, None]
        z_lcl = np.array([4000.0])  # LCL above all levels
        # min_depth=1 so contig_depth=0 (nothing above LCL) fails the threshold
        result = _lfc_index(z, b, z_lcl, min_depth=1.0)
        assert result[0] == 0

# extra_outputs tests
# ---------------------------------------------------------------------------


def _unstable_1col():
    """Return (p, zh, t, r) for the 'unstable' case as (nz, 1) arrays."""
    data = CapeCinData()
    return (
        data.p["unstable"][:, None],
        data.zh["unstable"][:, None],
        data.t["unstable"][:, None],
        data.r["unstable"][:, None],
    )


def test_extra_outputs_none_returns_two_tuple():
    """When extra_outputs is not set the return value must be a 2-tuple."""
    p, zh, t, r = _unstable_1col()
    result = thermo.cape_cin(p, zh, t, r, "surface")
    assert len(result) == 2, "Expected (cape, cin) 2-tuple when extra_outputs is None"
    cape, cin = result
    assert cape.shape == (1,)
    assert cin.shape == (1,)


def test_extra_outputs_empty_list_returns_two_tuple():
    """An empty extra_outputs list must behave the same as None."""
    p, zh, t, r = _unstable_1col()
    result = thermo.cape_cin(p, zh, t, r, "surface", extra_outputs=[])
    assert len(result) == 2


def test_extra_outputs_parcel_path_shape():
    """parcel_path arrays must have (nz, ...) shape; key levels must have (...) shape."""
    from earthkit.meteo.thermo.array.cape_cin import ParcelPath

    p, zh, t, r = _unstable_1col()
    nz = p.shape[0]
    horizontal_shape = (1,)

    cape, cin, extras = thermo.cape_cin(p, zh, t, r, "surface", extra_outputs=["parcel_path"])
    path = extras["parcel_path"]

    assert isinstance(path, ParcelPath)

    # Profile arrays
    for attr in ("p", "zh", "t", "r", "tv", "tv_env"):
        arr = getattr(path, attr)
        assert arr.shape == (nz,) + horizontal_shape, (
            f"parcel_path.{attr}: expected shape {(nz,) + horizontal_shape}, got {arr.shape}"
        )

    # Key-level arrays (horizontal only)
    for level_attr in ("lcl", "lfc", "el"):
        level = getattr(path, level_attr)
        assert level.p.shape == horizontal_shape, (
            f"parcel_path.{level_attr}.p: expected shape {horizontal_shape}, got {level.p.shape}"
        )
        assert level.t.shape == horizontal_shape, (
            f"parcel_path.{level_attr}.t: expected shape {horizontal_shape}, got {level.t.shape}"
        )
        assert level.zh.shape == horizontal_shape, (
            f"parcel_path.{level_attr}.zh: expected shape {horizontal_shape}, got {level.zh.shape}"
        )

    # Parcel origin
    assert path.origin.p.shape == horizontal_shape
    assert path.origin.t.shape == horizontal_shape
    assert path.origin.r.shape == horizontal_shape


def test_extra_outputs_parcel_path_nd_shape():
    """parcel_path shapes must generalise to arbitrary horizontal dimensions."""
    from earthkit.meteo.thermo.array.cape_cin import ParcelPath

    data = CapeCinData()
    nz = data.p_stacked.shape[0]
    ny, nx = 2, 2
    p = data.p_stacked.reshape(nz, ny, nx)
    t = data.t_stacked.reshape(nz, ny, nx)
    r = data.r_stacked.reshape(nz, ny, nx)
    zh = data.zh_stacked.reshape(nz, ny, nx)

    cape, cin, extras = thermo.cape_cin(p, zh, t, r, "surface", extra_outputs=["parcel_path"])
    path = extras["parcel_path"]

    assert isinstance(path, ParcelPath)
    for attr in ("p", "zh", "t", "r", "tv", "tv_env"):
        assert getattr(path, attr).shape == (nz, ny, nx), f"profile array {attr} wrong shape"
    for level_attr in ("lcl", "lfc", "el"):
        assert getattr(path, level_attr).p.shape == (ny, nx), f"key level {level_attr}.p wrong shape"
        assert getattr(path, level_attr).zh.shape == (ny, nx), f"key level {level_attr}.zh wrong shape"


def test_extra_outputs_standalone_key_levels():
    """Requesting 'lcl', 'lfc', 'el', 'parcel' individually returns the right objects."""
    from earthkit.meteo.thermo.array.cape_cin import ParcelOrigin, PressureLevel

    p, zh, t, r = _unstable_1col()
    cape, cin, extras = thermo.cape_cin(p, zh, t, r, "surface", extra_outputs=["lcl", "lfc", "el", "parcel"])
    assert set(extras.keys()) == {"lcl", "lfc", "el", "parcel"}

    for key in ("lcl", "lfc", "el"):
        assert isinstance(extras[key], PressureLevel), f"extras['{key}'] should be a PressureLevel"
        assert extras[key].p.shape == (1,)
        assert extras[key].t.shape == (1,)
        assert extras[key].zh.shape == (1,)

    assert isinstance(extras["parcel"], ParcelOrigin)
    assert extras["parcel"].p.shape == (1,)
    assert extras["parcel"].t.shape == (1,)
    assert extras["parcel"].r.shape == (1,)


def test_extra_outputs_key_levels_consistent_with_parcel_path():
    """Standalone lcl/lfc/el must match those embedded in parcel_path."""
    p, zh, t, r = _unstable_1col()
    cape, cin, extras = thermo.cape_cin(
        p, zh, t, r, "surface", extra_outputs=["lcl", "lfc", "el", "parcel", "parcel_path"]
    )
    path = extras["parcel_path"]

    np.testing.assert_array_equal(extras["lcl"].p, path.lcl.p)
    np.testing.assert_array_equal(extras["lcl"].t, path.lcl.t)
    np.testing.assert_array_equal(extras["lcl"].zh, path.lcl.zh)
    np.testing.assert_array_equal(extras["lfc"].p, path.lfc.p)
    np.testing.assert_array_equal(extras["lfc"].zh, path.lfc.zh)
    np.testing.assert_array_equal(extras["el"].p, path.el.p)
    np.testing.assert_array_equal(extras["el"].zh, path.el.zh)
    np.testing.assert_array_equal(extras["parcel"].p, path.origin.p)
    np.testing.assert_array_equal(extras["parcel"].r, path.origin.r)


def test_extra_outputs_cape_cin_values_unchanged():
    """extra_outputs must not alter the cape/cin values."""
    p, zh, t, r = _unstable_1col()
    cape_base, cin_base = thermo.cape_cin(p, zh, t, r, "surface")
    cape_ext, cin_ext, _ = thermo.cape_cin(p, zh, t, r, "surface", extra_outputs=["parcel_path"])
    np.testing.assert_array_equal(cape_base, cape_ext)
    np.testing.assert_array_equal(cin_base, cin_ext)


def test_extra_outputs_invalid_key_raises():
    """An unrecognised key in extra_outputs must raise ValueError."""
    p, zh, t, r = _unstable_1col()
    with pytest.raises(ValueError, match="extra_outputs"):
        thermo.cape_cin(p, zh, t, r, "surface", extra_outputs=["parcel_path", "not_a_real_key"])


def test_extra_outputs_parcel_path_pressure_sorted():
    """parcel_path.p must be in ascending order even when input is descending."""
    p, zh, t, r = _unstable_1col()
    p_flip = np.flip(p, axis=0)
    t_flip = np.flip(t, axis=0)
    zh_flip = np.flip(zh, axis=0)
    r_flip = np.flip(r, axis=0)

    _, _, extras = thermo.cape_cin(p_flip, zh_flip, t_flip, r_flip, "surface", extra_outputs=["parcel_path"])
    path_p = extras["parcel_path"].p[:, 0]
    assert np.all(np.diff(path_p) >= 0), "parcel_path.p must be sorted ascending"


def test_extra_outputs_vertical_axis_minus_1():
    """parcel_path profile arrays must have the vertical axis restored to the caller's position."""
    p, zh, t, r = _unstable_1col()
    # transpose to (1, nz) — vertical axis is now axis 1 (== -1)
    p_T, zh_T, t_T, r_T = p.T, zh.T, t.T, r.T

    cape, cin, extras = thermo.cape_cin(p_T, zh_T, t_T, r_T, "surface", vertical_axis=-1, extra_outputs=["parcel_path"])
    # profile arrays must mirror the caller's shape: (1, nz)
    nz = p.shape[0]
    for attr in ("p", "zh", "t", "r", "tv", "tv_env"):
        arr = getattr(extras["parcel_path"], attr)
        assert arr.shape == (1, nz), f"parcel_path.{attr}: expected (1, {nz}), got {arr.shape}"


def test_extra_outputs_vertical_axis_arbitrary():
    """parcel_path profile arrays must restore any non-zero vertical_axis."""
    data = CapeCinData()
    nz = data.p_stacked.shape[0]
    ny, nx = 2, 2
    # build (ny, nx, nz) input — vertical axis is 2
    p = data.p_stacked.reshape(nz, ny, nx)
    t = data.t_stacked.reshape(nz, ny, nx)
    r = data.r_stacked.reshape(nz, ny, nx)
    zh = data.zh_stacked.reshape(nz, ny, nx)
    # move vertical axis to position 2: shape becomes (ny, nx, nz)
    p_v2 = np.moveaxis(p, 0, 2)
    t_v2 = np.moveaxis(t, 0, 2)
    r_v2 = np.moveaxis(r, 0, 2)
    zh_v2 = np.moveaxis(zh, 0, 2)

    cape, cin, extras = thermo.cape_cin(
        p_v2, zh_v2, t_v2, r_v2, "surface", vertical_axis=2, extra_outputs=["parcel_path"]
    )
    # profile arrays must have shape (ny, nx, nz)
    for attr in ("p", "zh", "t", "r", "tv", "tv_env"):
        arr = getattr(extras["parcel_path"], attr)
        assert arr.shape == (ny, nx, nz), f"parcel_path.{attr}: expected {(ny, nx, nz)}, got {arr.shape}"
    # horizontal outputs are unaffected
    assert extras["parcel_path"].lcl.p.shape == (ny, nx)
