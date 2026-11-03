# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Tests for freezing_level and snowfall_limit (heights where a profile falls below a threshold)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from earthkit.meteo import thermo

xr = pytest.importorskip("xarray")

T0 = 273.15  # 0 degC (K)
HEIGHTS = [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0]  # bottom-up (m)


def _profiles(columns, heights=HEIGHTS):
    """Profiles of shape (z, column) from a list of bottom-up columns, with the heights broadcast."""
    values = xr.DataArray(np.asarray(columns, dtype=float).T, dims=("z", "column"))
    z = xr.DataArray(np.asarray(heights, dtype=float), dims="z").broadcast_like(values)
    return values, z


def _column_values(*columns):
    return xr.DataArray(np.asarray(columns, dtype=float), dims="column")


def _reference(values, heights, threshold, value_sfc=None, z_sfc=None, lapse_rate=None, min_height=None):
    """Independent per-column implementation.

    Scans the bottom-up profile top-down for the first layer with value >= threshold
    below and < threshold above.
    """
    out = np.full(values.shape[1], np.nan)
    for i in range(values.shape[1]):
        points = list(zip(heights[:, i], values[:, i]))
        if z_sfc is not None:
            points = [(z_sfc[i], value_sfc[i])] + [(h, v) for h, v in points if h > z_sfc[i]]
        if np.isnan(points).any():
            continue
        for (h0, v0), (h1, v1) in reversed(list(zip(points[:-1], points[1:]))):
            if v0 >= threshold > v1:
                out[i] = h0 + (threshold - v0) / (v1 - v0) * (h1 - h0)
                break
        else:
            if lapse_rate is not None:
                out[i] = z_sfc[i] - (threshold - value_sfc[i]) / lapse_rate
        if min_height is not None:
            out[i] = max(out[i], min_height)
    return out


@pytest.mark.parametrize("order", ["bottom_up", "top_down"])
def test_freezing_level_linear_profile(order):
    # T = 15 degC at sea level decreasing by 6.5 K/km: 0 degC at 15 / 0.0065 m
    z = np.arange(0.0, 10001.0, 500.0)
    t, z = _profiles([T0 + 15.0 - 0.0065 * z], heights=z)
    if order == "top_down":
        t, z = t[::-1], z[::-1]

    result = thermo.freezing_level(t, z)

    assert result.dims == ("column",)
    assert_allclose(result.values, [15.0 / 0.0065])


@pytest.mark.parametrize(
    "column,expected",
    [
        ([283, 280, 277, 274, 271, 268, 262], 2141.6667),  # single crossing
        ([278, 272, 276, 271, 270, 268, 262], 1785.0),  # inversion: top of the upper warm layer
        ([270, 275, 276, 272, 270, 268, 262], 1856.25),  # cold surface, warm layer aloft
        ([276, T0, 276, 274, 271, 268, 262], 2141.6667),  # touching 0 degC without crossing below
        ([280, 275, T0, 270, 268, 265, 260], 1500.0),  # crossing exactly on a level
        ([270, 268, 266, 264, 262, 260, 255], np.nan),  # colder than 0 degC everywhere
        ([270, 272, 274, 276, 278, 279, 280], np.nan),  # warmer at the top: only a crossing from below
        ([280, 278, 276, 275, 274, 274, T0], np.nan),  # top exactly at 0 degC: not below it
    ],
)
def test_freezing_level_crossing(column, expected):
    t, z = _profiles([column])
    assert_allclose(thermo.freezing_level(t, z).values, [expected], rtol=1e-6)


def test_freezing_level_matches_reference_on_random_profiles():
    rng = np.random.default_rng(0)
    n = 500
    heights = np.sort(rng.uniform(0.0, 9000.0, (12, n)), axis=0)
    z_sfc = rng.uniform(-50.0, 3000.0, n)
    t_sfc = rng.uniform(255.0, 300.0, n)
    # lapse-rate profiles with random warm layers, and arbitrary values below the surface
    t = t_sfc - rng.uniform(0.004, 0.009, n) * (heights - z_sfc)
    t += rng.uniform(0.0, 8.0, n) * np.exp(-(((heights - z_sfc - rng.uniform(0.0, 3000.0, n)) / 400.0) ** 2))
    t = np.where(heights > z_sfc, t, rng.uniform(250.0, 300.0, t.shape))

    da = lambda x: xr.DataArray(x, dims=("z", "column")[-x.ndim :])  # noqa: E731
    kwargs = dict(lapse_rate=0.0065, min_height=-100.0)
    result = thermo.freezing_level(da(t), da(heights), t_sfc=da(t_sfc), z_sfc=da(z_sfc), **kwargs)

    expected = _reference(t, heights, T0, t_sfc, z_sfc, **kwargs)
    assert_allclose(result.values, expected, rtol=1e-10)
    # the cases of interest all occur
    no_crossing = np.isnan(_reference(t, heights, T0, t_sfc, z_sfc))
    assert 0 < no_crossing.sum() < n
    assert (result.values == -100.0).any()


def test_snowfall_limit_default_threshold():
    # wet-bulb temperature decreasing by 5 K/km, 1.3 degC at 1250 m
    z = np.array(HEIGHTS)
    tw, z = _profiles([T0 + 1.3 - 0.005 * (z - 1250.0)], heights=z)

    assert_allclose(thermo.snowfall_limit(tw, z).values, [1250.0])


def test_snowfall_limit_and_freezing_level_differ_only_by_threshold():
    t, z = _profiles([[283, 280, 277, 274, 271, 268, 262], [278, 272, 276, 271, 270, 268, 262]])

    assert_allclose(
        thermo.snowfall_limit(t, z, threshold=T0).values,
        thermo.freezing_level(t, z).values,
    )
    assert_allclose(
        thermo.snowfall_limit(t, z).values,
        thermo.freezing_level(t, z, threshold=T0 + 1.3).values,
    )


def test_surface_point_crossing_below_lowest_level():
    # warm surface at 100 m, lowest level already below 0 degC: crossing in between
    t, z = _profiles([[272, 268, 265, 262, 260, 258, 250]])

    result = thermo.freezing_level(t, z, t_sfc=_column_values(275.0), z_sfc=_column_values(100.0))

    assert_allclose(result.values, [100.0 + (275.0 - T0) / (275.0 - 272.0) * 400.0])


def test_surface_value_at_threshold():
    # surface exactly at 0 degC and colder above: the freezing level is at the surface,
    # also when the underground levels are replaced by the surface point
    t, z = _profiles([[290, 272, 268, 265, 262, 260, 255]])

    result = thermo.freezing_level(t, z, t_sfc=_column_values(T0), z_sfc=_column_values(700.0))

    assert_allclose(result.values, [700.0])


def test_levels_below_surface_are_ignored():
    # surface at 1200 m: the warm levels at 500 m and 1000 m are underground
    t, z = _profiles([[290, 285, 272, 268, 265, 262, 255]])
    t_sfc, z_sfc = _column_values(274.0), _column_values(1200.0)

    result = thermo.freezing_level(t, z, t_sfc=t_sfc, z_sfc=z_sfc)
    expected = 1200.0 + (274.0 - T0) / (274.0 - 272.0) * 300.0

    assert_allclose(result.values, [expected])
    # without the surface, the underground levels give another crossing
    assert thermo.freezing_level(t, z).values[0] != pytest.approx(expected)


def test_levels_with_missing_height_are_below_surface():
    # NaN heights mark levels as underground, e.g. pressure levels with p >= surface pressure
    t, z = _profiles([[290, 274, 272, 268, 265, 262, 255]])
    z_masked = z.where(xr.DataArray([False, True, True, True, True, True, True], dims="z"))
    kwargs = dict(t_sfc=_column_values(276.0), z_sfc=_column_values(300.0))

    result = thermo.freezing_level(t, z_masked, **kwargs)

    assert_allclose(result.values, [1000.0 + (274.0 - T0) / (274.0 - 272.0) * 500.0])


@pytest.mark.parametrize(
    "column,t_sfc,expected",
    [
        ([270, 268, 266, 264, 262, 260, 255], 270.55, 200.0 - 2.6 / 0.0065),  # cold column: below the surface
        ([270, 272, 274, 276, 278, 279, 280], 270.55, 200.0 - 2.6 / 0.0065),  # no crossing from above
        ([283, 280, 277, 274, 271, 268, 262], 285.0, 2141.6667),  # crossing found: no extrapolation
    ],
)
def test_lapse_rate_extrapolation(column, t_sfc, expected):
    t, z = _profiles([column])

    result = thermo.freezing_level(t, z, t_sfc=_column_values(t_sfc), z_sfc=_column_values(200.0), lapse_rate=0.0065)

    assert_allclose(result.values, [expected], rtol=1e-6)


def test_lapse_rate_extrapolation_above_surface():
    # truncated profile warmer than the threshold everywhere: extrapolated above the surface
    t, z = _profiles([[280, 279, 278, 277, 276, 275, 274]])

    result = thermo.freezing_level(t, z, t_sfc=_column_values(281.0), z_sfc=_column_values(0.0), lapse_rate=0.005)

    assert_allclose(result.values, [(281.0 - T0) / 0.005])


def test_min_height():
    t, z = _profiles([[270, 268, 266, 264, 262, 260, 255], [283, 280, 277, 274, 271, 268, 262]])
    kwargs = dict(t_sfc=_column_values(265.0, 285.0), z_sfc=_column_values(0.0, 0.0), lapse_rate=0.0065)

    unbounded = thermo.freezing_level(t, z, **kwargs)
    bounded = thermo.freezing_level(t, z, min_height=-100.0, **kwargs)

    assert unbounded.values[0] < -100.0
    assert_allclose(bounded.values, [-100.0, unbounded.values[1]])


def test_missing_values():
    t, z = _profiles([
        [283, 280, np.nan, 274, 271, 268, 262],  # missing above the surface: NaN
        [np.nan, 280, 277, 274, 271, 268, 262],  # missing below the surface: ignored
        [270, 268, 266, 264, 262, 260, 255],  # complete
    ])
    kwargs = dict(t_sfc=_column_values(285.0, 285.0, 271.85), z_sfc=_column_values(0.0, 700.0, 0.0), lapse_rate=0.0065)

    result = thermo.freezing_level(t, z, **kwargs)

    assert np.isnan(result.values[0])
    assert_allclose(result.values[1:], [2141.6667, -200.0], rtol=1e-6)


def test_scalar_surface_values():
    t, z = _profiles([[272, 268, 265, 262, 260, 258, 250], [271, 268, 265, 262, 260, 258, 250]])

    result = thermo.freezing_level(t, z, t_sfc=275.0, z_sfc=100.0)

    assert_allclose(result.values, [100.0 + (275.0 - T0) / (275.0 - c) * (500.0 - 100.0) for c in (272.0, 271.0)])


def test_dims_and_coords():
    # (level, lat, lon) with a pressure coordinate on the vertical dimension
    rng = np.random.default_rng(1)
    level, lat, lon = [1000, 850, 700, 500, 300], [46.0, 47.0, 48.0], [7.0, 8.0]
    heights = np.array([100.0, 1500.0, 3000.0, 5600.0, 9200.0])[:, None, None]
    t_values = 288.0 - 0.0065 * heights + rng.uniform(-3.0, 3.0, (5, 3, 2))
    coords = {"level": level, "lat": lat, "lon": lon}
    t = xr.DataArray(t_values, dims=("level", "lat", "lon"), coords=coords, name="t", attrs={"units": "K"})
    z = xr.DataArray(np.broadcast_to(heights, t.shape), dims=t.dims, coords=coords)
    t_sfc = t.isel(level=0, drop=True) + 1.0
    z_sfc = xr.full_like(t_sfc, 50.0)

    result = thermo.freezing_level(t, z, t_sfc=t_sfc, z_sfc=z_sfc, vertical_dim="level")

    assert result.dims == ("lat", "lon")
    assert result.coords["lat"].values.tolist() == lat
    assert result.coords["lon"].values.tolist() == lon
    assert "level" not in result.coords
    assert result.name is None and result.attrs == {}
    expected = _reference(
        t_values.reshape(5, -1),
        np.broadcast_to(heights, t.shape).reshape(5, -1),
        T0,
        t_sfc.values.ravel(),
        z_sfc.values.ravel(),
    )
    assert_allclose(result.values.ravel(), expected)


def test_xarray_implementation_is_dispatched():
    from earthkit.meteo.thermo import xarray as thermo_xarray

    t, z = _profiles([[283, 280, 277, 274, 271, 268, 262]])

    assert_allclose(thermo.freezing_level(t, z).values, thermo_xarray.freezing_level(t, z).values)
    assert_allclose(thermo.snowfall_limit(t, z).values, thermo_xarray.snowfall_limit(t, z).values)


@pytest.mark.parametrize("func", [thermo.freezing_level, thermo.snowfall_limit])
@pytest.mark.parametrize(
    "value_sfc,z_sfc,lapse_rate,match",
    [
        (None, 0.0, None, "given together"),
        (280.0, None, None, "given together"),
        (None, None, 0.0065, "requires the surface"),
        (280.0, 0.0, 0.0, "must be positive"),
        (280.0, 0.0, -0.0065, "must be positive"),
    ],
)
def test_invalid_arguments(func, value_sfc, z_sfc, lapse_rate, match):
    t, z = _profiles([[283, 280, 277, 274, 271, 268, 262]])

    with pytest.raises(ValueError, match=match):
        func(t, z, value_sfc, z_sfc, lapse_rate=lapse_rate)
