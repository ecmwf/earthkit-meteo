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
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", name)


def read_data_file(path):
    return np.genfromtxt(data_file(path), delimiter=",", names=True)


def save_cape_cin_reference(expected_data):
    """Regenerate the CSV reference files from dicts of 1-D arrays."""
    np.savetxt(
        data_file("cape_cin_expected.csv"),
        np.column_stack(list(expected_data.values())),
        delimiter=",",
        header=",".join(expected_data.keys()),
    )


class CapeCinData:
    """Load CAPE/CIN test data from the reference CSV files.

    The last row of each case in the input CSV is the surface level.
    The input CSV stores mixing ratio; we convert to specific humidity
    using :func:`specific_humidity_from_mixing_ratio` so that the tests
    exercise the public API which accepts specific humidity.

    Grid arrays have shape ``(nz, ...)`` and surface arrays have shape ``(...)``.

    Attributes
    ----------
    p, t, zh, q : dict[str, np.ndarray]
        Per-case grid arrays (shape ``(nz,)``), keyed by case name.
    p_sfc, t_sfc, zh_sfc, q_sfc : dict[str, np.ndarray]
        Per-case surface scalars, keyed by case name.
    p_stacked, t_stacked, zh_stacked, q_stacked : np.ndarray
        All four cases stacked column-wise (shape ``(nz, ncases)``).
    p_sfc_stacked, t_sfc_stacked, zh_sfc_stacked, q_sfc_stacked : np.ndarray
        Surface values for all cases (shape ``(ncases,)``).
    expected_cape, expected_cin : dict[str, np.ndarray]
        Per-parcel-type arrays of expected values (shape ``(ncases,)``).
    """

    def __init__(self):
        raw = read_data_file("cape_cin_input.csv")
        expected = read_data_file("cape_cin_expected.csv")

        # Split the last row (surface) from the grid.
        # The CSV stores mixing ratio; convert to specific humidity (q).
        self.p = {n: raw[f"{n}_p"][:-1] for n in CASE_NAMES}
        self.t = {n: raw[f"{n}_t"][:-1] for n in CASE_NAMES}
        self.zh = {n: raw[f"{n}_zh"][:-1] for n in CASE_NAMES}
        self.q = {n: thermo.specific_humidity_from_mixing_ratio(raw[f"{n}_r"][:-1]) for n in CASE_NAMES}

        self.p_sfc = {n: raw[f"{n}_p"][-1] for n in CASE_NAMES}
        self.t_sfc = {n: raw[f"{n}_t"][-1] for n in CASE_NAMES}
        self.zh_sfc = {n: raw[f"{n}_zh"][-1] for n in CASE_NAMES}
        self.q_sfc = {n: thermo.specific_humidity_from_mixing_ratio(raw[f"{n}_r"][-1]) for n in CASE_NAMES}

        self.p_stacked = np.column_stack([self.p[n] for n in CASE_NAMES])
        self.t_stacked = np.column_stack([self.t[n] for n in CASE_NAMES])
        self.zh_stacked = np.column_stack([self.zh[n] for n in CASE_NAMES])
        self.q_stacked = np.column_stack([self.q[n] for n in CASE_NAMES])

        self.p_sfc_stacked = np.array([self.p_sfc[n] for n in CASE_NAMES])
        self.t_sfc_stacked = np.array([self.t_sfc[n] for n in CASE_NAMES])
        self.zh_sfc_stacked = np.array([self.zh_sfc[n] for n in CASE_NAMES])
        self.q_sfc_stacked = np.array([self.q_sfc[n] for n in CASE_NAMES])

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
    q = data.q[case_name][:, None]
    p_sfc = np.atleast_1d(data.p_sfc[case_name])
    zh_sfc = np.atleast_1d(data.zh_sfc[case_name])
    t_sfc = np.atleast_1d(data.t_sfc[case_name])
    q_sfc = np.atleast_1d(data.q_sfc[case_name])

    cape, cin = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, parcel_type)

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
            data.p_stacked,
            data.zh_stacked,
            data.t_stacked,
            data.q_stacked,
            data.p_sfc_stacked,
            data.zh_sfc_stacked,
            data.t_sfc_stacked,
            data.q_sfc_stacked,
            parcel_type,
        )
        np.testing.assert_allclose(cape, data.expected_cape[parcel_type], atol=1)
        np.testing.assert_allclose(cin, data.expected_cin[parcel_type], atol=1)


def test_cape_cin_vertical_axis_minus_1():
    data = CapeCinData()
    p = data.p_stacked.T
    t = data.t_stacked.T
    q = data.q_stacked.T
    zh = data.zh_stacked.T

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(
            p,
            zh,
            t,
            q,
            data.p_sfc_stacked,
            data.zh_sfc_stacked,
            data.t_sfc_stacked,
            data.q_sfc_stacked,
            parcel_type,
            vertical_axis=-1,
        )
        np.testing.assert_allclose(cape, data.expected_cape[parcel_type], atol=1)
        np.testing.assert_allclose(cin, data.expected_cin[parcel_type], atol=1)


def test_cape_cin_arbitrary_nd_shape():
    nz = 3
    horizontal_shape = (2, 3, 4)
    shape = (nz,) + horizontal_shape

    p = np.broadcast_to(np.array([80000, 90000, 100000])[:, None, None, None], shape)
    t = np.broadcast_to(np.array([270, 280, 290])[:, None, None, None], shape)
    q = np.broadcast_to(np.array([0.03, 0.02, 0.01])[:, None, None, None], shape)
    zh = np.broadcast_to(np.array([300, 200, 100])[:, None, None, None], shape)

    p_sfc = np.full(horizontal_shape, 101000.0)
    zh_sfc = np.full(horizontal_shape, 50.0)
    t_sfc = np.full(horizontal_shape, 292.0)
    q_sfc = np.full(horizontal_shape, 0.008)

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, parcel_type)
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
    p_sfc = data.p_sfc_stacked.reshape(ny, nx)
    zh_sfc = data.zh_sfc_stacked.reshape(ny, nx)
    t_sfc = data.t_sfc_stacked.reshape(ny, nx)
    q_sfc = data.q_sfc_stacked.reshape(ny, nx)

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, parcel_type)
        np.testing.assert_allclose(cape, data.expected_cape[parcel_type].reshape(ny, nx), atol=1)
        np.testing.assert_allclose(cin, data.expected_cin[parcel_type].reshape(ny, nx), atol=1)


def test_cape_cin_options_forwarded():
    """Regression: options passed to cape_cin() must reach the subclass."""
    data = CapeCinData()
    p = data.p["unstable"][:, None]
    t = data.t["unstable"][:, None]
    q = data.q["unstable"][:, None]
    zh = data.zh["unstable"][:, None]
    p_sfc = np.atleast_1d(data.p_sfc["unstable"])
    zh_sfc = np.atleast_1d(data.zh_sfc["unstable"])
    t_sfc = np.atleast_1d(data.t_sfc["unstable"])
    q_sfc = np.atleast_1d(data.q_sfc["unstable"])

    # Default layer_depth (5000 Pa) vs a wider mixed layer (15000 Pa) must differ.
    cape_default, _ = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "mixed")
    cape_wide, _ = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "mixed", layer_depth=15000)
    assert not np.isclose(cape_default, cape_wide, atol=1), (
        "layer_depth option was not forwarded to the mixed-layer parcel computation"
    )


def test_cape_cin_invalid_parcel_type():
    """cape_cin() must raise ValueError for an unrecognised parcel_type."""
    data = CapeCinData()
    p = data.p["unstable"][:, None]
    t = data.t["unstable"][:, None]
    q = data.q["unstable"][:, None]
    zh = data.zh["unstable"][:, None]
    p_sfc = np.atleast_1d(data.p_sfc["unstable"])
    zh_sfc = np.atleast_1d(data.zh_sfc["unstable"])
    t_sfc = np.atleast_1d(data.t_sfc["unstable"])
    q_sfc = np.atleast_1d(data.q_sfc["unstable"])

    with pytest.raises(ValueError, match="parcel_type"):
        thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "unknown_parcel")


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

    p = np.hstack([np.full_like(p_real, np.nan), p_real])
    t = np.hstack([np.full_like(t_real, np.nan), t_real])
    zh = np.hstack([np.full_like(zh_real, np.nan), zh_real])
    q = np.hstack([np.full_like(q_real, np.nan), q_real])

    p_sfc = np.array([np.nan, data.p_sfc["stable"]])
    zh_sfc = np.array([np.nan, data.zh_sfc["stable"]])
    t_sfc = np.array([np.nan, data.t_sfc["stable"]])
    q_sfc = np.array([np.nan, data.q_sfc["stable"]])

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, parcel_type)
        assert np.isnan(cape[0]), f"{parcel_type}: expected NaN for all-NaN column, got {cape[0]}"
        assert np.isnan(cin[0]), f"{parcel_type}: expected NaN for all-NaN column, got {cin[0]}"
        np.testing.assert_allclose(cape[1], 0.0, atol=1)
        np.testing.assert_allclose(cin[1], 0.0, atol=1)


def test_cape_cin_no_lfc():
    """A strongly stable (isothermal) profile must return CAPE=0, CIN=0."""
    data = CapeCinData()
    p = data.p["stable"][:, None]
    zh = data.zh["stable"][:, None]
    q = data.q["stable"][:, None]
    # Isothermal profile: parcel lifted dry-adiabatically is always cooler than env.
    t = np.full_like(p, 260.0)
    p_sfc = np.atleast_1d(data.p_sfc["stable"])
    zh_sfc = np.atleast_1d(data.zh_sfc["stable"])
    t_sfc = np.atleast_1d(260.0)
    q_sfc = np.atleast_1d(data.q_sfc["stable"])

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, parcel_type)
        np.testing.assert_allclose(cape, 0.0, atol=1e-6, err_msg=f"{parcel_type}: expected CAPE=0")
        np.testing.assert_allclose(cin, 0.0, atol=1e-6, err_msg=f"{parcel_type}: expected CIN=0")


def test_cape_cin_unsorted_pressure():
    """A vertically flipped (surface-first) profile must give the same result as sorted."""
    data = CapeCinData()
    p = data.p["unstable"][:, None]
    t = data.t["unstable"][:, None]
    zh = data.zh["unstable"][:, None]
    q = data.q["unstable"][:, None]
    p_sfc = np.atleast_1d(data.p_sfc["unstable"])
    zh_sfc = np.atleast_1d(data.zh_sfc["unstable"])
    t_sfc = np.atleast_1d(data.t_sfc["unstable"])
    q_sfc = np.atleast_1d(data.q_sfc["unstable"])

    p_flip = np.flip(p, axis=0)
    t_flip = np.flip(t, axis=0)
    zh_flip = np.flip(zh, axis=0)
    q_flip = np.flip(q, axis=0)

    for parcel_type in PARCEL_TYPES:
        cape_sorted, cin_sorted = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, parcel_type)
        cape_flip, cin_flip = thermo.cape_cin(p_flip, zh_flip, t_flip, q_flip, p_sfc, zh_sfc, t_sfc, q_sfc, parcel_type)
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
    q = data.q["unstable"][:, None]
    p_sfc_1d = np.atleast_1d(data.p_sfc["unstable"])
    zh_sfc_1d = np.atleast_1d(data.zh_sfc["unstable"])
    t_sfc_1d = np.atleast_1d(data.t_sfc["unstable"])
    q_sfc_1d = np.atleast_1d(data.q_sfc["unstable"])

    p_flip = np.flip(p, axis=0)
    t_flip = np.flip(t, axis=0)
    zh_flip = np.flip(zh, axis=0)
    q_flip = np.flip(q, axis=0)

    # col0 = flipped (unsorted), col1 = sorted — both should give the same result
    p2 = np.hstack([p_flip, p])
    t2 = np.hstack([t_flip, t])
    zh2 = np.hstack([zh_flip, zh])
    q2 = np.hstack([q_flip, q])
    p_sfc2 = np.concatenate([p_sfc_1d, p_sfc_1d])
    zh_sfc2 = np.concatenate([zh_sfc_1d, zh_sfc_1d])
    t_sfc2 = np.concatenate([t_sfc_1d, t_sfc_1d])
    q_sfc2 = np.concatenate([q_sfc_1d, q_sfc_1d])

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p2, zh2, t2, q2, p_sfc2, zh_sfc2, t_sfc2, q_sfc2, parcel_type)
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
    p_sfc = np.atleast_1d(data.p_sfc["stable"])
    zh_sfc = np.atleast_1d(data.zh_sfc["stable"])
    t_sfc = np.atleast_1d(data.t_sfc["stable"])
    q_sfc = np.atleast_1d(1e-9)

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, parcel_type)
        np.testing.assert_allclose(cape, 0.0, atol=1, err_msg=f"{parcel_type}: expected CAPE=0 for dry profile")
        np.testing.assert_allclose(cin, 0.0, atol=1, err_msg=f"{parcel_type}: expected CIN=0 for dry profile")


# ---------------------------------------------------------------------------
# extra_outputs tests
# ---------------------------------------------------------------------------


def _unstable_1col():
    """Return (p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc) for the 'unstable' case.

    Grid arrays have shape (nz, 1), surface arrays have shape (1,).
    """
    data = CapeCinData()
    return (
        data.p["unstable"][:, None],
        data.zh["unstable"][:, None],
        data.t["unstable"][:, None],
        data.q["unstable"][:, None],
        np.atleast_1d(data.p_sfc["unstable"]),
        np.atleast_1d(data.zh_sfc["unstable"]),
        np.atleast_1d(data.t_sfc["unstable"]),
        np.atleast_1d(data.q_sfc["unstable"]),
    )


def test_extra_outputs_none_returns_two_tuple():
    """When extra_outputs is not set the return value must be a 2-tuple."""
    p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc = _unstable_1col()
    result = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "surface")
    assert len(result) == 2, "Expected (cape, cin) 2-tuple when extra_outputs is None"
    cape, cin = result
    assert cape.shape == (1,)
    assert cin.shape == (1,)


def test_extra_outputs_empty_list_returns_two_tuple():
    """An empty extra_outputs list must behave the same as None."""
    p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc = _unstable_1col()
    result = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "surface", extra_outputs=[])
    assert len(result) == 2


def test_extra_outputs_parcel_path_shape():
    """parcel_path arrays must have (nz_total, ...) shape; key levels must have (...) shape."""
    from earthkit.meteo.thermo.array.cape_cin import ParcelPath

    p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc = _unstable_1col()
    nz_grid = p.shape[0]
    nz_total = nz_grid + 1  # grid + concatenated surface
    horizontal_shape = (1,)

    cape, cin, extras = thermo.cape_cin(
        p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "surface", extra_outputs=["parcel_path"]
    )
    path = extras["parcel_path"]

    assert isinstance(path, ParcelPath)

    # Profile arrays
    for attr in ("p", "t", "q", "tv", "tv_env"):
        arr = getattr(path, attr)
        assert arr.shape == (nz_total,) + horizontal_shape, (
            f"parcel_path.{attr}: expected shape {(nz_total,) + horizontal_shape}, got {arr.shape}"
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

    # Parcel origin
    assert path.origin.p.shape == horizontal_shape
    assert path.origin.t.shape == horizontal_shape
    assert path.origin.q.shape == horizontal_shape


def test_extra_outputs_parcel_path_nd_shape():
    """parcel_path shapes must generalise to arbitrary horizontal dimensions."""
    from earthkit.meteo.thermo.array.cape_cin import ParcelPath

    data = CapeCinData()
    nz = data.p_stacked.shape[0]
    nz_total = nz + 1  # grid + concatenated surface
    ny, nx = 2, 2
    p = data.p_stacked.reshape(nz, ny, nx)
    t = data.t_stacked.reshape(nz, ny, nx)
    q = data.q_stacked.reshape(nz, ny, nx)
    zh = data.zh_stacked.reshape(nz, ny, nx)
    p_sfc = data.p_sfc_stacked.reshape(ny, nx)
    zh_sfc = data.zh_sfc_stacked.reshape(ny, nx)
    t_sfc = data.t_sfc_stacked.reshape(ny, nx)
    q_sfc = data.q_sfc_stacked.reshape(ny, nx)

    cape, cin, extras = thermo.cape_cin(
        p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "surface", extra_outputs=["parcel_path"]
    )
    path = extras["parcel_path"]

    assert isinstance(path, ParcelPath)
    for attr in ("p", "t", "q", "tv", "tv_env"):
        assert getattr(path, attr).shape == (nz_total, ny, nx), f"profile array {attr} wrong shape"
    for level_attr in ("lcl", "lfc", "el"):
        assert getattr(path, level_attr).p.shape == (ny, nx), f"key level {level_attr}.p wrong shape"


def test_extra_outputs_standalone_key_levels():
    """Requesting 'lcl', 'lfc', 'el', 'parcel' individually returns the right objects."""
    from earthkit.meteo.thermo.array.cape_cin import ParcelOrigin, PressureLevel

    p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc = _unstable_1col()
    cape, cin, extras = thermo.cape_cin(
        p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "surface", extra_outputs=["lcl", "lfc", "el", "parcel"]
    )
    assert set(extras.keys()) == {"lcl", "lfc", "el", "parcel"}

    for key in ("lcl", "lfc", "el"):
        assert isinstance(extras[key], PressureLevel), f"extras['{key}'] should be a PressureLevel"
        assert extras[key].p.shape == (1,)
        assert extras[key].t.shape == (1,)

    assert isinstance(extras["parcel"], ParcelOrigin)
    assert extras["parcel"].p.shape == (1,)
    assert extras["parcel"].t.shape == (1,)
    assert extras["parcel"].q.shape == (1,)


def test_extra_outputs_key_levels_consistent_with_parcel_path():
    """Standalone lcl/lfc/el must match those embedded in parcel_path."""
    p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc = _unstable_1col()
    cape, cin, extras = thermo.cape_cin(
        p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "surface", extra_outputs=["lcl", "lfc", "el", "parcel", "parcel_path"]
    )
    path = extras["parcel_path"]

    np.testing.assert_array_equal(extras["lcl"].p, path.lcl.p)
    np.testing.assert_array_equal(extras["lcl"].t, path.lcl.t)
    np.testing.assert_array_equal(extras["lfc"].p, path.lfc.p)
    np.testing.assert_array_equal(extras["el"].p, path.el.p)
    np.testing.assert_array_equal(extras["parcel"].p, path.origin.p)
    np.testing.assert_array_equal(extras["parcel"].q, path.origin.q)


def test_extra_outputs_cape_cin_values_unchanged():
    """extra_outputs must not alter the cape/cin values."""
    p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc = _unstable_1col()
    cape_base, cin_base = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "surface")
    cape_ext, cin_ext, _ = thermo.cape_cin(
        p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "surface", extra_outputs=["parcel_path"]
    )
    np.testing.assert_array_equal(cape_base, cape_ext)
    np.testing.assert_array_equal(cin_base, cin_ext)


def test_extra_outputs_invalid_key_raises():
    """An unrecognised key in extra_outputs must raise ValueError."""
    p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc = _unstable_1col()
    with pytest.raises(ValueError, match="extra_outputs"):
        thermo.cape_cin(
            p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "surface", extra_outputs=["parcel_path", "not_a_real_key"]
        )


def test_extra_outputs_parcel_path_pressure_sorted():
    """parcel_path.p must be in ascending order even when input is descending."""
    p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc = _unstable_1col()
    p_flip = np.flip(p, axis=0)
    t_flip = np.flip(t, axis=0)
    zh_flip = np.flip(zh, axis=0)
    q_flip = np.flip(q, axis=0)

    _, _, extras = thermo.cape_cin(
        p_flip, zh_flip, t_flip, q_flip, p_sfc, zh_sfc, t_sfc, q_sfc, "surface", extra_outputs=["parcel_path"]
    )
    path_p = extras["parcel_path"].p[:, 0]
    # Filter out NaN (subground) levels for the sorted check
    valid_p = path_p[~np.isnan(path_p)]
    assert np.all(np.diff(valid_p) >= 0), "parcel_path.p must be sorted ascending (ignoring NaN)"


def test_extra_outputs_vertical_axis_minus_1():
    """parcel_path profile arrays must have the vertical axis restored to the caller's position."""
    p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc = _unstable_1col()
    # transpose to (1, nz) — vertical axis is now axis 1 (== -1)
    p_T, zh_T, t_T, q_T = p.T, zh.T, t.T, q.T

    cape, cin, extras = thermo.cape_cin(
        p_T, zh_T, t_T, q_T, p_sfc, zh_sfc, t_sfc, q_sfc, "surface", vertical_axis=-1, extra_outputs=["parcel_path"]
    )
    # profile arrays must mirror the caller's shape: (1, nz_total)
    nz_total = p.shape[0] + 1  # grid + surface
    assert extras["parcel_path"].p.shape == (1, nz_total), (
        f"Expected (1, {nz_total}), got {extras['parcel_path'].p.shape}"
    )


def test_extra_outputs_vertical_axis_arbitrary():
    """parcel_path profile arrays must restore any non-zero vertical_axis."""
    data = CapeCinData()
    nz = data.p_stacked.shape[0]
    nz_total = nz + 1  # grid + surface
    ny, nx = 2, 2
    # build (ny, nx, nz) input — vertical axis is 2
    p = data.p_stacked.reshape(nz, ny, nx)
    t = data.t_stacked.reshape(nz, ny, nx)
    q = data.q_stacked.reshape(nz, ny, nx)
    zh = data.zh_stacked.reshape(nz, ny, nx)
    p_sfc = data.p_sfc_stacked.reshape(ny, nx)
    zh_sfc = data.zh_sfc_stacked.reshape(ny, nx)
    t_sfc = data.t_sfc_stacked.reshape(ny, nx)
    q_sfc = data.q_sfc_stacked.reshape(ny, nx)
    # move vertical axis to position 2: shape becomes (ny, nx, nz)
    p_v2 = np.moveaxis(p, 0, 2)
    t_v2 = np.moveaxis(t, 0, 2)
    q_v2 = np.moveaxis(q, 0, 2)
    zh_v2 = np.moveaxis(zh, 0, 2)

    cape, cin, extras = thermo.cape_cin(
        p_v2, zh_v2, t_v2, q_v2, p_sfc, zh_sfc, t_sfc, q_sfc, "surface", vertical_axis=2, extra_outputs=["parcel_path"]
    )
    # profile arrays must have shape (ny, nx, nz_total)
    for attr in ("p", "t", "q", "tv", "tv_env"):
        arr = getattr(extras["parcel_path"], attr)
        assert arr.shape == (ny, nx, nz_total), f"parcel_path.{attr}: expected {(ny, nx, nz_total)}, got {arr.shape}"
    # horizontal outputs are unaffected
    assert extras["parcel_path"].lcl.p.shape == (ny, nx)


# ---------------------------------------------------------------------------
# Subground masking tests
# ---------------------------------------------------------------------------


def test_cape_cin_subground_levels_masked():
    """Grid levels below the surface height must be masked and not affect the result.

    Use the 'unstable' case. Add two extra grid levels with zh below the
    surface and bogus temperature values. The result must match the original
    profile without the extra levels.
    """
    data = CapeCinData()
    p = data.p["unstable"][:, None]
    t = data.t["unstable"][:, None]
    zh = data.zh["unstable"][:, None]
    q = data.q["unstable"][:, None]
    p_sfc = np.atleast_1d(data.p_sfc["unstable"])
    zh_sfc = np.atleast_1d(data.zh_sfc["unstable"])
    t_sfc = np.atleast_1d(data.t_sfc["unstable"])
    q_sfc = np.atleast_1d(data.q_sfc["unstable"])

    # Reference result without subground levels
    cape_ref, cin_ref = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "surface")

    # Add two subground levels (below zh_sfc) with bogus values
    p_extra = np.array([[105000.0], [110000.0]])
    zh_extra = np.array([[-50.0], [-200.0]])  # well below surface
    t_extra = np.array([[350.0], [400.0]])  # bogus — should be masked
    q_extra = np.array([[0.1], [0.2]])  # bogus

    p_aug = np.concatenate([p, p_extra], axis=0)
    zh_aug = np.concatenate([zh, zh_extra], axis=0)
    t_aug = np.concatenate([t, t_extra], axis=0)
    q_aug = np.concatenate([q, q_extra], axis=0)

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p_aug, zh_aug, t_aug, q_aug, p_sfc, zh_sfc, t_sfc, q_sfc, parcel_type)
        cape_ref, cin_ref = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, parcel_type)
        np.testing.assert_allclose(
            cape,
            cape_ref,
            atol=1,
            err_msg=f"{parcel_type}: subground levels altered CAPE",
        )
        np.testing.assert_allclose(
            cin,
            cin_ref,
            atol=1,
            err_msg=f"{parcel_type}: subground levels altered CIN",
        )


def test_cape_cin_subground_nans_not_propagated():
    """NaN values at sub-ground positions must NOT make the output NaN.

    Input grids may already contain NaN at sub-ground levels. These NaN values
    are expected and should be silently masked, not treated as bad data.
    """
    data = CapeCinData()
    p = data.p["unstable"][:, None].copy()
    t = data.t["unstable"][:, None].copy()
    zh = data.zh["unstable"][:, None].copy()
    q = data.q["unstable"][:, None].copy()
    p_sfc = np.atleast_1d(data.p_sfc["unstable"])
    zh_sfc = np.atleast_1d(data.zh_sfc["unstable"])
    t_sfc = np.atleast_1d(data.t_sfc["unstable"])
    q_sfc = np.atleast_1d(data.q_sfc["unstable"])

    # Add a subground level that is already NaN in the input
    p_extra = np.array([[110000.0]])
    zh_extra = np.array([[-100.0]])  # below surface
    t_extra = np.array([[np.nan]])
    q_extra = np.array([[np.nan]])

    p_aug = np.concatenate([p, p_extra], axis=0)
    zh_aug = np.concatenate([zh, zh_extra], axis=0)
    t_aug = np.concatenate([t, t_extra], axis=0)
    q_aug = np.concatenate([q, q_extra], axis=0)

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p_aug, zh_aug, t_aug, q_aug, p_sfc, zh_sfc, t_sfc, q_sfc, parcel_type)
        assert not np.isnan(cape[0]), f"{parcel_type}: subground NaN wrongly propagated to CAPE"
        assert not np.isnan(cin[0]), f"{parcel_type}: subground NaN wrongly propagated to CIN"


def test_cape_cin_baddata_nan_still_propagated():
    """NaN at an above-ground grid level must still produce NaN output."""
    data = CapeCinData()
    p = data.p["unstable"][:, None].copy()
    t = data.t["unstable"][:, None].copy()
    zh = data.zh["unstable"][:, None].copy()
    q = data.q["unstable"][:, None].copy()
    p_sfc = np.atleast_1d(data.p_sfc["unstable"])
    zh_sfc = np.atleast_1d(data.zh_sfc["unstable"])
    t_sfc = np.atleast_1d(data.t_sfc["unstable"])
    q_sfc = np.atleast_1d(data.q_sfc["unstable"])

    # Introduce a NaN at an above-ground level
    t[5, 0] = np.nan

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, parcel_type)
        assert np.isnan(cape[0]), f"{parcel_type}: above-ground NaN did not propagate to CAPE"
        assert np.isnan(cin[0]), f"{parcel_type}: above-ground NaN did not propagate to CIN"


def test_cape_cin_stacked_different_surfaces():
    """Stacked columns with different surface heights each mask different levels.

    Column 0: surface at zh=1080 m (elevated_instability) — has 2 subground levels.
    Column 1: surface at zh=-0.26 m (stable) — no subground levels.
    Both must produce valid (non-NaN) results.
    """
    data = CapeCinData()
    p = np.column_stack([data.p["elevated_instability"], data.p["stable"]])
    zh = np.column_stack([data.zh["elevated_instability"], data.zh["stable"]])
    t = np.column_stack([data.t["elevated_instability"], data.t["stable"]])
    q = np.column_stack([data.q["elevated_instability"], data.q["stable"]])
    p_sfc = np.array([data.p_sfc["elevated_instability"], data.p_sfc["stable"]])
    zh_sfc = np.array([data.zh_sfc["elevated_instability"], data.zh_sfc["stable"]])
    t_sfc = np.array([data.t_sfc["elevated_instability"], data.t_sfc["stable"]])
    q_sfc = np.array([data.q_sfc["elevated_instability"], data.q_sfc["stable"]])

    for parcel_type in PARCEL_TYPES:
        cape, cin = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, parcel_type)
        assert not np.any(np.isnan(cape)), f"{parcel_type}: unexpected NaN in CAPE"
        assert not np.any(np.isnan(cin)), f"{parcel_type}: unexpected NaN in CIN"


def test_cape_cin_mixed_uses_surface_pressure():
    """Mixed parcel must use p_sfc as the bottom bound, not max grid pressure.

    For the 'elevated_instability' case the surface is at 89048 Pa but the
    grid includes a 100000 Pa level. The mixed-layer average should be
    computed over the surface-to-(surface - layer_depth) range, not over
    the 100000 Pa level.
    """
    data = CapeCinData()
    p = data.p["elevated_instability"][:, None]
    zh = data.zh["elevated_instability"][:, None]
    t = data.t["elevated_instability"][:, None]
    q = data.q["elevated_instability"][:, None]
    p_sfc = np.atleast_1d(data.p_sfc["elevated_instability"])
    zh_sfc = np.atleast_1d(data.zh_sfc["elevated_instability"])
    t_sfc = np.atleast_1d(data.t_sfc["elevated_instability"])
    q_sfc = np.atleast_1d(data.q_sfc["elevated_instability"])

    _, _, extras = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "mixed", extra_outputs=["parcel"])
    # The parcel departure pressure must equal the surface pressure
    np.testing.assert_allclose(extras["parcel"].p, p_sfc, atol=1)


def test_cape_cin_surface_parcel_uses_surface_inputs():
    """Surface parcel must use the explicit surface inputs, not a grid level."""
    data = CapeCinData()
    p = data.p["elevated_instability"][:, None]
    zh = data.zh["elevated_instability"][:, None]
    t = data.t["elevated_instability"][:, None]
    q = data.q["elevated_instability"][:, None]
    p_sfc = np.atleast_1d(data.p_sfc["elevated_instability"])
    zh_sfc = np.atleast_1d(data.zh_sfc["elevated_instability"])
    t_sfc = np.atleast_1d(data.t_sfc["elevated_instability"])
    q_sfc = np.atleast_1d(data.q_sfc["elevated_instability"])

    _, _, extras = thermo.cape_cin(p, zh, t, q, p_sfc, zh_sfc, t_sfc, q_sfc, "surface", extra_outputs=["parcel"])
    np.testing.assert_allclose(extras["parcel"].p, p_sfc)
    np.testing.assert_allclose(extras["parcel"].t, t_sfc)
    np.testing.assert_allclose(extras["parcel"].q, q_sfc)
