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


def _cape_cin_func(parcel_type):
    """Return the appropriate top-level cape/cin function for the given parcel type."""
    return {
        "surface": thermo.surface_cape_cin,
        "mixed": thermo.mixed_layer_cape_cin,
        "mu": thermo.most_unstable_cape_cin,
    }[parcel_type]


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
    p, t, zh, q : dict[str, np.ndarray]
        Per-case 1-D arrays (shape ``(nz,)``), keyed by case name.
        Pressure is in Pa, temperature in K, height in m, specific humidity in kg/kg.
    p_stacked, t_stacked, zh_stacked, q_stacked : np.ndarray
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
        self.q = {n: input[f"{n}_q"] for n in CASE_NAMES}

        self.p_stacked = np.column_stack([self.p[n] for n in CASE_NAMES])
        self.t_stacked = np.column_stack([self.t[n] for n in CASE_NAMES])
        self.zh_stacked = np.column_stack([self.zh[n] for n in CASE_NAMES])
        self.q_stacked = np.column_stack([self.q[n] for n in CASE_NAMES])

        self.expected_cape = {parcel_type: expected[f"{parcel_type}_cape"] for parcel_type in PARCEL_TYPES}
        self.expected_cin = {parcel_type: expected[f"{parcel_type}_cin"] for parcel_type in PARCEL_TYPES}


def _sfc_from_profile(p, t, q, zh):
    """Return (p_sfc, t_sfc, q_sfc, zh_sfc) taken from the bottom level of a stacked profile.

    The reference CSV stores the surface as the last row of each column. The new
    cape_cin API expects pressure-level inputs (without the surface) and the
    surface arrays as separate horizontal-only arguments.
    """
    return p[-1], t[-1], q[-1], zh[-1]


def _strip_sfc(p, t, q, zh):
    """Strip the bottom (surface) level from stacked profile arrays."""
    return p[:-1], t[:-1], q[:-1], zh[:-1]


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
    q = data.q[case_name][:, None]

    func = _cape_cin_func(parcel_type)
    cape, cin = func(*_strip_sfc(p, t, q, zh), *_sfc_from_profile(p, t, q, zh))

    case_idx = CASE_NAMES.index(case_name)
    np.testing.assert_allclose(cape, data.expected_cape[parcel_type][case_idx], atol=1)
    np.testing.assert_allclose(cin, data.expected_cin[parcel_type][case_idx], atol=1)


# ---------------------------------------------------------------------------
# Multi-profile (stacked) and shape tests
# ---------------------------------------------------------------------------


def test_cape_cin_stacked():
    data = CapeCinData()
    p, zh, t, q = data.p_stacked, data.zh_stacked, data.t_stacked, data.q_stacked
    for parcel_type in PARCEL_TYPES:
        func = _cape_cin_func(parcel_type)
        cape, cin = func(*_strip_sfc(p, t, q, zh), *_sfc_from_profile(p, t, q, zh))
        np.testing.assert_allclose(cape, data.expected_cape[parcel_type], atol=1)
        np.testing.assert_allclose(cin, data.expected_cin[parcel_type], atol=1)


def test_cape_cin_vertical_axis_minus_1():
    data = CapeCinData()
    # Strip the surface (last row of the stacked profile) before transposing.
    p = data.p_stacked[:-1].T
    t = data.t_stacked[:-1].T
    q = data.q_stacked[:-1].T
    zh = data.zh_stacked[:-1].T
    # Surface: last level of the original (non-transposed) stacked profile
    p_sfc = data.p_stacked[-1, :]
    t_sfc = data.t_stacked[-1, :]
    q_sfc = data.q_stacked[-1, :]
    zh_sfc = data.zh_stacked[-1, :]

    for parcel_type in PARCEL_TYPES:
        func = _cape_cin_func(parcel_type)
        cape, cin = func(p, t, q, zh, p_sfc, t_sfc, q_sfc, zh_sfc, vertical_axis=-1)
        np.testing.assert_allclose(cape, data.expected_cape[parcel_type], atol=1)
        np.testing.assert_allclose(cin, data.expected_cin[parcel_type], atol=1)


def test_cape_cin_arbitrary_nd_shape():
    nz = 3
    horizontal_shape = (2, 3, 4)
    shape = (nz,) + horizontal_shape

    p = np.broadcast_to(np.array([100000, 90000, 80000])[:, None, None, None], shape)
    t = np.broadcast_to(np.array([290, 280, 270])[:, None, None, None], shape)
    q = np.broadcast_to(np.array([0.01, 0.02, 0.03])[:, None, None, None], shape)
    zh = np.broadcast_to(np.array([100, 200, 300])[:, None, None, None], shape)

    for parcel_type in PARCEL_TYPES:
        func = _cape_cin_func(parcel_type)
        cape, cin = func(*_strip_sfc(p, t, q, zh), *_sfc_from_profile(p, t, q, zh))
        assert cape.shape == horizontal_shape
        assert cin.shape == horizontal_shape


def test_cape_cin_lat_lon():
    data = CapeCinData()
    nz = data.p_stacked.shape[0]
    ny, nx = 2, 2
    p = data.p_stacked.reshape(nz, ny, nx)
    t = data.t_stacked.reshape(nz, ny, nx)
    q = data.q_stacked.reshape(nz, ny, nx)
    zh = data.zh_stacked.reshape(nz, ny, nx)

    for parcel_type in PARCEL_TYPES:
        func = _cape_cin_func(parcel_type)
        cape, cin = func(*_strip_sfc(p, t, q, zh), *_sfc_from_profile(p, t, q, zh))
        np.testing.assert_allclose(cape, data.expected_cape[parcel_type].reshape(ny, nx), atol=1)
        np.testing.assert_allclose(cin, data.expected_cin[parcel_type].reshape(ny, nx), atol=1)


def test_cape_cin_missing_values():
    data = CapeCinData()
    # Extract surface (last row) before stripping it from the PL input
    p_sfc = data.p_stacked[-1, :]
    t_sfc = data.t_stacked[-1, :]
    q_sfc = data.q_stacked[-1, :]
    zh_sfc = data.zh_stacked[-1, :]

    p = data.p_stacked[:-1].copy()
    t = data.t_stacked[:-1].copy()
    q = data.q_stacked[:-1].copy()
    zh = data.zh_stacked[:-1].copy()

    # Introduce NaNs at above-ground levels: column 0 via t, column 2 via q
    t[0, 0] = np.nan
    q[3, 2] = np.nan

    for parcel_type in PARCEL_TYPES:
        func = _cape_cin_func(parcel_type)
        cape, cin = func(p, t, q, zh, p_sfc, t_sfc, q_sfc, zh_sfc)
        assert np.isnan(cape[0])
        assert np.isnan(cin[0])
        assert np.isnan(cape[2])
        assert np.isnan(cin[2])


def test_mixed_layer_cape_cin_layer_depth_forwarded():
    """Regression: layer_depth must reach the mixed-layer parcel computation."""
    data = CapeCinData()
    p = data.p["unstable"][:, None]
    t = data.t["unstable"][:, None]
    q = data.q["unstable"][:, None]
    zh = data.zh["unstable"][:, None]

    cape_default, _ = thermo.mixed_layer_cape_cin(*_strip_sfc(p, t, q, zh), *_sfc_from_profile(p, t, q, zh))
    cape_wide, _ = thermo.mixed_layer_cape_cin(
        *_strip_sfc(p, t, q, zh), *_sfc_from_profile(p, t, q, zh), layer_depth=15000
    )
    assert not np.isclose(cape_default, cape_wide, atol=1), (
        "layer_depth option was not forwarded to the mixed-layer parcel computation"
    )


def _synthetic_elevated_mu_profile():
    """Return a synthetic profile whose most-unstable parcel sits at zh_agl=500 m.

    The 500 m level has a much higher specific humidity than the surface, so its
    equivalent potential temperature is the largest of the column.
    """
    p_sfc = np.array([100000.0])
    t_sfc = np.array([280.0])
    q_sfc = np.array([0.005])
    zh_sfc = np.array([0.0])

    p_pl = np.array([[97000.0], [94000.0], [90000.0], [85000.0], [70000.0], [50000.0]])
    zh_pl = np.array([[500.0], [1000.0], [1500.0], [2500.0], [4000.0], [6000.0]])
    t_pl = np.array([[277.0], [274.0], [270.0], [262.0], [248.0], [230.0]])
    q_pl = np.array([[0.015], [0.010], [0.008], [0.005], [0.003], [0.001]])
    return p_pl, t_pl, q_pl, zh_pl, p_sfc, t_sfc, q_sfc, zh_sfc


def test_most_unstable_cape_cin_max_search_height_forwarded():
    """Max_search_height must reach the most-unstable parcel computation.

    Synthetic profile: the most-unstable parcel lives at 500 m above the
    surface, so capping ``max_search_height`` below 500 m forces the surface
    parcel to be selected and changes the CAPE value.
    """
    args = _synthetic_elevated_mu_profile()

    cape_low, _ = thermo.most_unstable_cape_cin(*args, max_search_height=400)
    cape_high, _ = thermo.most_unstable_cape_cin(*args, max_search_height=1500)
    assert not np.isclose(cape_low, cape_high, atol=1), (
        "max_search_height option was not forwarded to the most-unstable parcel computation"
    )


def test_most_unstable_cape_cin_exclude_surface_layer():
    """exclude_surface_layer must exclude the surface parcel from the
    candidate set so that the selected parcel comes from above the surface.

    Constructed so that the surface parcel would otherwise be the most
    unstable: when ``exclude_surface_layer=True`` the next-best (elevated)
    parcel is chosen instead, yielding a different CAPE value.
    """
    p_sfc = np.array([100000.0])
    t_sfc = np.array([295.0])
    q_sfc = np.array([0.018])  # very moist surface → highest theta_ep
    zh_sfc = np.array([0.0])

    p_pl = np.array([[97000.0], [94000.0], [90000.0], [85000.0], [70000.0], [50000.0]])
    zh_pl = np.array([[500.0], [1000.0], [1500.0], [2500.0], [4000.0], [6000.0]])
    t_pl = np.array([[290.0], [285.0], [280.0], [272.0], [258.0], [240.0]])
    q_pl = np.array([[0.012], [0.010], [0.008], [0.005], [0.003], [0.001]])

    cape_default, _ = thermo.most_unstable_cape_cin(p_pl, t_pl, q_pl, zh_pl, p_sfc, t_sfc, q_sfc, zh_sfc)
    cape_excl, _ = thermo.most_unstable_cape_cin(
        p_pl, t_pl, q_pl, zh_pl, p_sfc, t_sfc, q_sfc, zh_sfc, exclude_surface_layer=True
    )
    assert not np.isclose(cape_default, cape_excl, atol=1), (
        "exclude_surface_layer option did not affect the most-unstable parcel selection"
    )


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
    q_real = data.q["stable"][:, None]

    # Bottom row is the surface; PL inputs exclude it.
    p_real_pl = p_real[:-1]
    t_real_pl = t_real[:-1]
    zh_real_pl = zh_real[:-1]
    q_real_pl = q_real[:-1]

    p = np.hstack([np.full_like(p_real_pl, np.nan), p_real_pl])
    t = np.hstack([np.full_like(t_real_pl, np.nan), t_real_pl])
    zh = np.hstack([np.full_like(zh_real_pl, np.nan), zh_real_pl])
    q = np.hstack([np.full_like(q_real_pl, np.nan), q_real_pl])

    # Surface for column 0 is NaN (profile is all NaN); column 1 uses real bottom level
    p_sfc = np.array([np.nan, p_real[-1, 0]])
    t_sfc = np.array([np.nan, t_real[-1, 0]])
    q_sfc = np.array([np.nan, q_real[-1, 0]])
    zh_sfc = np.array([np.nan, zh_real[-1, 0]])

    for parcel_type in PARCEL_TYPES:
        func = _cape_cin_func(parcel_type)
        cape, cin = func(p, t, q, zh, p_sfc, t_sfc, q_sfc, zh_sfc)
        assert np.isnan(cape[0]), f"{parcel_type}: expected NaN for all-NaN column, got {cape[0]}"
        assert np.isnan(cin[0]), f"{parcel_type}: expected NaN for all-NaN column, got {cin[0]}"
        np.testing.assert_allclose(cape[1], 0.0, atol=1)
        np.testing.assert_allclose(cin[1], 0.0, atol=1)


def test_cape_cin_no_lfc():
    """A strongly stable (isothermal) profile must return CAPE=0, CIN=0."""
    data = CapeCinData()
    p_full = data.p["stable"][:, None]
    zh_full = data.zh["stable"][:, None]
    q_full = data.q["stable"][:, None]
    # PL-only inputs (strip surface = bottom row)
    p = p_full[:-1]
    zh = zh_full[:-1]
    q = q_full[:-1]
    # Isothermal profile: parcel lifted dry-adiabatically is always cooler than env.
    t = np.full_like(p, 260.0)
    t_sfc = np.array([260.0])

    for parcel_type in PARCEL_TYPES:
        func = _cape_cin_func(parcel_type)
        cape, cin = func(p, t, q, zh, p_full[-1], t_sfc, q_full[-1], zh_full[-1])
        np.testing.assert_allclose(cape, 0.0, atol=1e-6, err_msg=f"{parcel_type}: expected CAPE=0")
        np.testing.assert_allclose(cin, 0.0, atol=1e-6, err_msg=f"{parcel_type}: expected CIN=0")


def test_cape_cin_unsorted_pressure():
    """A vertically flipped (surface-first) profile must give the same result as sorted."""
    data = CapeCinData()
    p_full = data.p["unstable"][:, None]
    t_full = data.t["unstable"][:, None]
    zh_full = data.zh["unstable"][:, None]
    q_full = data.q["unstable"][:, None]

    # PL-only inputs (strip surface = bottom row)
    p = p_full[:-1]
    t = t_full[:-1]
    zh = zh_full[:-1]
    q = q_full[:-1]

    p_flip = np.flip(p, axis=0)
    t_flip = np.flip(t, axis=0)
    zh_flip = np.flip(zh, axis=0)
    q_flip = np.flip(q, axis=0)

    for parcel_type in PARCEL_TYPES:
        # Both sorted and flipped arrays share the same surface
        p_sfc = p_full[-1]
        t_sfc = t_full[-1]
        q_sfc = q_full[-1]
        zh_sfc = zh_full[-1]
        func = _cape_cin_func(parcel_type)
        cape_sorted, cin_sorted = func(p, t, q, zh, p_sfc, t_sfc, q_sfc, zh_sfc)
        cape_flip, cin_flip = func(p_flip, t_flip, q_flip, zh_flip, p_sfc, t_sfc, q_sfc, zh_sfc)
        np.testing.assert_allclose(
            cape_flip, cape_sorted, atol=1, err_msg=f"{parcel_type}: CAPE differs for flipped input"
        )
        np.testing.assert_allclose(
            cin_flip, cin_sorted, atol=1, err_msg=f"{parcel_type}: CIN differs for flipped input"
        )


def test_cape_cin_unsorted_pressure_stacked():
    """Mixed stacked input (one sorted, one unsorted column) must sort both correctly."""
    data = CapeCinData()
    p_full = data.p["unstable"][:, None]
    t_full = data.t["unstable"][:, None]
    zh_full = data.zh["unstable"][:, None]
    q_full = data.q["unstable"][:, None]

    # PL-only inputs
    p = p_full[:-1]
    t = t_full[:-1]
    zh = zh_full[:-1]
    q = q_full[:-1]

    p_flip = np.flip(p, axis=0)
    t_flip = np.flip(t, axis=0)
    zh_flip = np.flip(zh, axis=0)
    q_flip = np.flip(q, axis=0)

    # col0 = flipped (unsorted), col1 = sorted — both should give the same result
    p2 = np.hstack([p_flip, p])
    t2 = np.hstack([t_flip, t])
    zh2 = np.hstack([zh_flip, zh])
    q2 = np.hstack([q_flip, q])
    # Surface (bottom level of original profile) is the same for both columns
    p_sfc2 = np.array([p_full[-1, 0], p_full[-1, 0]])
    t_sfc2 = np.array([t_full[-1, 0], t_full[-1, 0]])
    q_sfc2 = np.array([q_full[-1, 0], q_full[-1, 0]])
    zh_sfc2 = np.array([zh_full[-1, 0], zh_full[-1, 0]])

    for parcel_type in PARCEL_TYPES:
        func = _cape_cin_func(parcel_type)
        cape, cin = func(p2, t2, q2, zh2, p_sfc2, t_sfc2, q_sfc2, zh_sfc2)
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
    """A near-zero specific humidity profile must return CAPE=0, CIN=0 without raising."""
    data = CapeCinData()
    p = data.p["stable"][:, None]
    zh = data.zh["stable"][:, None]
    t = data.t["stable"][:, None]
    q = np.full_like(p, 1e-9)  # effectively bone-dry

    for parcel_type in PARCEL_TYPES:
        func = _cape_cin_func(parcel_type)
        cape, cin = func(*_strip_sfc(p, t, q, zh), *_sfc_from_profile(p, t, q, zh))
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
    """Return (p, zh, t, q) for the 'unstable' case as (nz, 1) arrays."""
    data = CapeCinData()
    return (
        data.p["unstable"][:, None],
        data.zh["unstable"][:, None],
        data.t["unstable"][:, None],
        data.q["unstable"][:, None],
    )


def test_extra_outputs_none_returns_two_tuple():
    """When extra_outputs is not set the return value must be a 2-tuple."""
    p, zh, t, q = _unstable_1col()
    result = thermo.surface_cape_cin(*_strip_sfc(p, t, q, zh), *_sfc_from_profile(p, t, q, zh))
    assert len(result) == 2, "Expected (cape, cin) 2-tuple when extra_outputs is None"
    cape, cin = result
    assert cape.shape == (1,)
    assert cin.shape == (1,)


def test_extra_outputs_empty_list_returns_two_tuple():
    """An empty extra_outputs list must behave the same as None."""
    p, zh, t, q = _unstable_1col()
    result = thermo.surface_cape_cin(*_strip_sfc(p, t, q, zh), *_sfc_from_profile(p, t, q, zh), extra_outputs=[])
    assert len(result) == 2


def test_extra_outputs_parcel_path_shape():
    """parcel_path arrays must have (nz_pl+1, ...) shape (PL + surface);
    key levels must have (...) shape.
    """
    from earthkit.meteo.thermo.array.cape_cin import ParcelPath

    p, zh, t, q = _unstable_1col()
    nz_pl = p.shape[0] - 1  # one row is the surface, the rest are pressure levels
    nz_path = nz_pl + 1  # parcel_path includes the surface as an additional level
    horizontal_shape = (1,)

    cape, cin, extras = thermo.surface_cape_cin(
        *_strip_sfc(p, t, q, zh), *_sfc_from_profile(p, t, q, zh), extra_outputs=["parcel_path"]
    )
    path = extras["parcel_path"]

    assert isinstance(path, ParcelPath)

    # Profile arrays
    for attr in ("zh_agl", "t", "q", "tv", "tv_env"):
        arr = getattr(path, attr)
        assert arr.shape == (nz_path,) + horizontal_shape, (
            f"parcel_path.{attr}: expected shape {(nz_path,) + horizontal_shape}, got {arr.shape}"
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
        assert level.zh_agl.shape == horizontal_shape, (
            f"parcel_path.{level_attr}.zh_agl: expected shape {horizontal_shape}, got {level.zh_agl.shape}"
        )

    # Parcel origin
    assert path.origin.p.shape == horizontal_shape
    assert path.origin.t.shape == horizontal_shape
    assert path.origin.q.shape == horizontal_shape


def test_extra_outputs_parcel_path_nd_shape():
    """parcel_path shapes must generalise to arbitrary horizontal dimensions."""
    from earthkit.meteo.thermo.array.cape_cin import ParcelPath

    data = CapeCinData()
    nz = data.p_stacked.shape[0]
    ny, nx = 2, 2
    p = data.p_stacked.reshape(nz, ny, nx)
    t = data.t_stacked.reshape(nz, ny, nx)
    q = data.q_stacked.reshape(nz, ny, nx)
    zh = data.zh_stacked.reshape(nz, ny, nx)

    cape, cin, extras = thermo.surface_cape_cin(
        *_strip_sfc(p, t, q, zh), *_sfc_from_profile(p, t, q, zh), extra_outputs=["parcel_path"]
    )
    path = extras["parcel_path"]

    # PL inputs have nz-1 levels; parcel_path includes the surface so it has nz levels.
    nz_path = nz

    assert isinstance(path, ParcelPath)
    for attr in ("zh_agl", "t", "q", "tv", "tv_env"):
        assert getattr(path, attr).shape == (nz_path, ny, nx), f"profile array {attr} wrong shape"
    for level_attr in ("lcl", "lfc", "el"):
        assert getattr(path, level_attr).p.shape == (ny, nx), f"key level {level_attr}.p wrong shape"
        assert getattr(path, level_attr).zh_agl.shape == (ny, nx), f"key level {level_attr}.zh_agl wrong shape"


def test_extra_outputs_standalone_key_levels():
    """Requesting 'lcl', 'lfc', 'el', 'parcel' individually returns the right objects."""
    from earthkit.meteo.thermo.array.cape_cin import ParcelLevel, ParcelOrigin

    p, zh, t, q = _unstable_1col()
    cape, cin, extras = thermo.surface_cape_cin(
        *_strip_sfc(p, t, q, zh),
        *_sfc_from_profile(p, t, q, zh),
        extra_outputs=["lcl", "lfc", "el", "parcel"],
    )
    assert set(extras.keys()) == {"lcl", "lfc", "el", "parcel"}

    for key in ("lcl", "lfc", "el"):
        assert isinstance(extras[key], ParcelLevel), f"extras['{key}'] should be a ParcelLevel"
        assert extras[key].p.shape == (1,)
        assert extras[key].t.shape == (1,)
        assert extras[key].zh_agl.shape == (1,)

    assert isinstance(extras["parcel"], ParcelOrigin)
    assert extras["parcel"].p.shape == (1,)
    assert extras["parcel"].t.shape == (1,)
    assert extras["parcel"].q.shape == (1,)


def test_extra_outputs_key_levels_consistent_with_parcel_path():
    """Standalone lcl/lfc/el must match those embedded in parcel_path."""
    p, zh, t, q = _unstable_1col()
    cape, cin, extras = thermo.surface_cape_cin(
        *_strip_sfc(p, t, q, zh),
        *_sfc_from_profile(p, t, q, zh),
        extra_outputs=["lcl", "lfc", "el", "parcel", "parcel_path"],
    )
    path = extras["parcel_path"]

    np.testing.assert_array_equal(extras["lcl"].p, path.lcl.p)
    np.testing.assert_array_equal(extras["lcl"].t, path.lcl.t)
    np.testing.assert_array_equal(extras["lcl"].zh_agl, path.lcl.zh_agl)
    np.testing.assert_array_equal(extras["lfc"].p, path.lfc.p)
    np.testing.assert_array_equal(extras["lfc"].zh_agl, path.lfc.zh_agl)
    np.testing.assert_array_equal(extras["el"].p, path.el.p)
    np.testing.assert_array_equal(extras["el"].zh_agl, path.el.zh_agl)
    np.testing.assert_array_equal(extras["parcel"].p, path.origin.p)
    np.testing.assert_array_equal(extras["parcel"].q, path.origin.q)


def test_extra_outputs_cape_cin_values_unchanged():
    """extra_outputs must not alter the cape/cin values."""
    p, zh, t, q = _unstable_1col()
    cape_base, cin_base = thermo.surface_cape_cin(*_strip_sfc(p, t, q, zh), *_sfc_from_profile(p, t, q, zh))
    cape_ext, cin_ext, _ = thermo.surface_cape_cin(
        *_strip_sfc(p, t, q, zh), *_sfc_from_profile(p, t, q, zh), extra_outputs=["parcel_path"]
    )
    np.testing.assert_array_equal(cape_base, cape_ext)
    np.testing.assert_array_equal(cin_base, cin_ext)


def test_extra_outputs_invalid_key_raises():
    """An unrecognised key in extra_outputs must raise ValueError."""
    p, zh, t, q = _unstable_1col()
    with pytest.raises(ValueError, match="extra_outputs"):
        thermo.surface_cape_cin(
            *_strip_sfc(p, t, q, zh),
            *_sfc_from_profile(p, t, q, zh),
            extra_outputs=["parcel_path", "not_a_real_key"],
        )


def test_extra_outputs_parcel_path_height_sorted():
    """parcel_path.z_agl must be in decending order even when input is ascending."""
    p, zh, t, q = _unstable_1col()
    p_pl, t_pl, q_pl, zh_pl = _strip_sfc(p, t, q, zh)
    p_flip = np.flip(p_pl, axis=0)
    t_flip = np.flip(t_pl, axis=0)
    zh_flip = np.flip(zh_pl, axis=0)
    q_flip = np.flip(q_pl, axis=0)

    _, _, extras = thermo.surface_cape_cin(
        p_flip,
        t_flip,
        q_flip,
        zh_flip,
        *_sfc_from_profile(p, t, q, zh),  # surface from original
        extra_outputs=["parcel_path"],
    )
    path_zh_agl = extras["parcel_path"].zh_agl[:, 0]
    finite_zh = path_zh_agl[np.isfinite(path_zh_agl)]
    assert np.all(np.diff(finite_zh) <= 0), (
        "parcel_path.zh_agl must be sorted descending (NaN sub-ground levels at end are excluded)"
    )


def test_extra_outputs_vertical_axis_minus_1():
    """parcel_path profile arrays must have the vertical axis restored to the caller's position."""
    p, zh, t, q = _unstable_1col()
    p_pl, t_pl, q_pl, zh_pl = _strip_sfc(p, t, q, zh)
    # transpose to (1, nz_pl) — vertical axis is now axis 1 (== -1)
    p_T, t_T, q_T, zh_T = p_pl.T, t_pl.T, q_pl.T, zh_pl.T
    # Surface is the bottom row of the original (nz, 1) profile, shape (1,)
    p_sfc, t_sfc, q_sfc, zh_sfc = _sfc_from_profile(p, t, q, zh)

    cape, cin, extras = thermo.surface_cape_cin(
        p_T, t_T, q_T, zh_T, p_sfc, t_sfc, q_sfc, zh_sfc, vertical_axis=-1, extra_outputs=["parcel_path"]
    )
    # profile arrays must mirror the caller's shape: (1, nz_pl + 1)
    nz_path = p_pl.shape[0] + 1
    for attr in ("zh_agl", "t", "q", "tv", "tv_env"):
        arr = getattr(extras["parcel_path"], attr)
        assert arr.shape == (1, nz_path), f"parcel_path.{attr}: expected (1, {nz_path}), got {arr.shape}"


def test_extra_outputs_vertical_axis_arbitrary():
    """parcel_path profile arrays must restore any non-zero vertical_axis."""
    data = CapeCinData()
    nz = data.p_stacked.shape[0]
    ny, nx = 2, 2
    # build (nz, ny, nx) full-stack input then strip the surface row
    p_full = data.p_stacked.reshape(nz, ny, nx)
    t_full = data.t_stacked.reshape(nz, ny, nx)
    q_full = data.q_stacked.reshape(nz, ny, nx)
    zh_full = data.zh_stacked.reshape(nz, ny, nx)

    p = p_full[:-1]
    t = t_full[:-1]
    q = q_full[:-1]
    zh = zh_full[:-1]
    # move vertical axis to position 2: shape becomes (ny, nx, nz_pl)
    p_v2 = np.moveaxis(p, 0, 2)
    t_v2 = np.moveaxis(t, 0, 2)
    q_v2 = np.moveaxis(q, 0, 2)
    zh_v2 = np.moveaxis(zh, 0, 2)

    cape, cin, extras = thermo.surface_cape_cin(
        p_v2,
        t_v2,
        q_v2,
        zh_v2,
        p_full[-1],
        t_full[-1],
        q_full[-1],
        zh_full[-1],
        vertical_axis=2,
        extra_outputs=["parcel_path"],
    )
    # parcel_path includes the surface, so along the vertical axis it has nz_pl + 1 = nz levels
    for attr in ("zh_agl", "t", "q", "tv", "tv_env"):
        arr = getattr(extras["parcel_path"], attr)
        assert arr.shape == (ny, nx, nz), f"parcel_path.{attr}: expected {(ny, nx, nz)}, got {arr.shape}"
    # horizontal outputs are unaffected
    assert extras["parcel_path"].lcl.p.shape == (ny, nx)


# ---------------------------------------------------------------------------
# Sub-ground level handling
# ---------------------------------------------------------------------------


def test_subground_levels_ignored():
    """Profile with extra sub-ground levels (p > p_sfc) must give the same CAPE/CIN
    as the same profile without those levels.
    """
    data = CapeCinData()
    p = data.p["unstable"][:, None]
    t = data.t["unstable"][:, None]
    zh = data.zh["unstable"][:, None]
    q = data.q["unstable"][:, None]

    p_sfc = p[-1]
    t_sfc = t[-1]
    q_sfc = q[-1]
    zh_sfc = zh[-1]

    # Append two sub-ground levels below the surface
    p_sub = np.array([p_sfc[0] + 500.0, p_sfc[0] + 1000.0])[:, None]
    t_sub = np.array([t_sfc[0] + 1.0, t_sfc[0] + 2.0])[:, None]
    q_sub = np.array([q_sfc[0] + 5e-4, q_sfc[0] + 1e-3])[:, None]
    zh_sub = np.array([zh_sfc[0] - 50.0, zh_sfc[0] - 100.0])[:, None]

    p_ext = np.vstack([p, p_sub])
    t_ext = np.vstack([t, t_sub])
    q_ext = np.vstack([q, q_sub])
    zh_ext = np.vstack([zh, zh_sub])

    for parcel_type in PARCEL_TYPES:
        func = _cape_cin_func(parcel_type)
        cape_base, cin_base = func(p, t, q, zh, p_sfc, t_sfc, q_sfc, zh_sfc)
        cape_ext, cin_ext = func(p_ext, t_ext, q_ext, zh_ext, p_sfc, t_sfc, q_sfc, zh_sfc)
        np.testing.assert_allclose(
            cape_ext,
            cape_base,
            atol=1,
            err_msg=f"{parcel_type}: CAPE differs when sub-ground levels are present",
        )
        np.testing.assert_allclose(
            cin_ext,
            cin_base,
            atol=1,
            err_msg=f"{parcel_type}: CIN differs when sub-ground levels are present",
        )


def test_nan_in_subground_does_not_propagate():
    """A NaN value at a sub-ground level must not cause NaN outputs."""
    data = CapeCinData()
    p = data.p["unstable"][:, None]
    t = data.t["unstable"][:, None]
    zh = data.zh["unstable"][:, None]
    q = data.q["unstable"][:, None]

    p_sfc = p[-1]
    t_sfc = t[-1]
    q_sfc = q[-1]
    zh_sfc = zh[-1]

    # Append a sub-ground level whose t and q values are NaN
    p_sub = np.array([p_sfc[0] + 500.0])[:, None]
    t_sub = np.array([np.nan])[:, None]
    q_sub = np.array([np.nan])[:, None]
    zh_sub = np.array([zh_sfc[0] - 50.0])[:, None]

    p_ext = np.vstack([p, p_sub])
    t_ext = np.vstack([t, t_sub])
    q_ext = np.vstack([q, q_sub])
    zh_ext = np.vstack([zh, zh_sub])

    for parcel_type in PARCEL_TYPES:
        func = _cape_cin_func(parcel_type)
        cape, cin = func(p_ext, t_ext, q_ext, zh_ext, p_sfc, t_sfc, q_sfc, zh_sfc)
        assert np.isfinite(cin[0]), f"{parcel_type}: CIN is NaN/Inf due to sub-ground NaN"


def test_surface_parcel_uses_sfc_values():
    """The surface parcel must be lifted from p_sfc/t_sfc/q_sfc, not the profile bottom."""
    data = CapeCinData()
    p = data.p["unstable"][:, None]
    t = data.t["unstable"][:, None]
    zh = data.zh["unstable"][:, None]
    q = data.q["unstable"][:, None]

    p_sfc = p[-1]
    q_sfc = q[-1]
    zh_sfc = zh[-1]
    t_sfc_default = t[-1]

    # Warmer surface → more CAPE
    t_sfc_warm = t_sfc_default + 3.0

    cape_default, _ = thermo.surface_cape_cin(p, t, q, zh, p_sfc, t_sfc_default, q_sfc, zh_sfc)
    cape_warm, _ = thermo.surface_cape_cin(p, t, q, zh, p_sfc, t_sfc_warm, q_sfc, zh_sfc)

    assert cape_warm[0] > cape_default[0], "Warmer surface temperature should produce more CAPE for the surface parcel"
