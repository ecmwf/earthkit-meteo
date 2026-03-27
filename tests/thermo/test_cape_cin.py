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

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

CASE_NAMES = ["stable", "elevated_instability", "unstable", "large_cape_small_cin"]
PARCEL_TYPES = ["surface", "mixed", "mu"]


def data_file(name):
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", name)


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
        cape, cin = thermo.cape_cin(
            data.p_stacked, data.zh_stacked, data.t_stacked, data.r_stacked, parcel_type
        )
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
    assert not np.isclose(
        cape_default, cape_wide, atol=1
    ), "layer_depth option was not forwarded to the mixed-layer parcel computation"


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
        np.testing.assert_allclose(
            cape, 0.0, atol=1, err_msg=f"{parcel_type}: expected CAPE=0 for dry profile"
        )
        np.testing.assert_allclose(cin, 0.0, atol=1, err_msg=f"{parcel_type}: expected CIN=0 for dry profile")
