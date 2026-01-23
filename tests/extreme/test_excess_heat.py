# (C) Copyright 2025 - ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from earthkit.meteo.extreme import excess_heat


class TestDailyMeanTemeratureArgDayStart:

    @pytest.fixture
    def t2m(self):
        nt = 4 * 24
        return xr.DataArray(
            data=np.arange(nt),
            dims=["valid_time"],
            coords={"valid_time": pd.date_range("2026-01-01", periods=nt, freq="1h")},
            name="t2m",
        )

    @pytest.mark.parametrize("start", [-5, 0, 3])
    def test_number_value_interpreted_as_hours(self, t2m, start):
        dmt_tdelta = excess_heat.daily_mean_temperature(t2m, day_start=np.timedelta64(start, "h"))
        dmt_number = excess_heat.daily_mean_temperature(t2m, day_start=start)
        np.testing.assert_allclose(dmt_tdelta, dmt_number)

    def test_zero_means_day_starts_at_midnight(self, t2m):
        dmt = excess_heat.daily_mean_temperature(t2m, day_start=0, time_shift=0)
        np.testing.assert_allclose(dmt, [0.5 * (0 + 23), 0.5 * (24 + 47), 0.5 * (48 + 71), 0.5 * (72 + 95)])

    @pytest.mark.parametrize("start", [-9, -5, -2])
    def test_negative_value_means_day_starts_early(self, t2m, start):
        dmt = excess_heat.daily_mean_temperature(t2m, day_start=start, time_shift=0)
        # Partial periods removed: days 2, 3, 4 returned; 1 and 5 partial
        np.testing.assert_allclose(
            dmt, [0.5 * (24 + 47) + start, 0.5 * (48 + 71) + start, 0.5 * (72 + 95) + start]
        )

    @pytest.mark.parametrize("start", [2, 5, 9])
    def test_positive_means_day_starts_late(self, t2m, start):
        dmt = excess_heat.daily_mean_temperature(t2m, day_start=start, time_shift=0)
        # Partial periods removed: days 1, 2, 3 returned; 0 and 4 partial
        np.testing.assert_allclose(
            dmt, [0.5 * (0 + 23) + start, 0.5 * (24 + 47) + start, 0.5 * (48 + 71) + start]
        )


class TestDailyMeanTemperatureArgTimeShift:

    @pytest.fixture
    def t2m(self):
        nx, nt = 3, 4 * 24
        return xr.DataArray(
            # Values differ by order of magnitude in space, increase over time
            data=np.logspace(0, nx - 1, nx)[None, :] * np.arange(nt)[:, None],
            dims=["valid_time", "x"],
            coords={
                "valid_time": pd.date_range("2026-01-01", periods=nt, freq="1h"),
                "x": np.arange(nx),
            },
            name="t2m",
        )

    @pytest.mark.parametrize("shift", [-5, 0, 3])
    def test_number_value_interpreted_as_hours(self, t2m, shift):
        dmt_tdelta = excess_heat.daily_mean_temperature(
            t2m, day_start=0, time_shift=np.timedelta64(shift, "h")
        )
        dmt_number = excess_heat.daily_mean_temperature(t2m, day_start=0, time_shift=shift)
        np.testing.assert_allclose(dmt_tdelta, dmt_number)

    def test_field_access_with_str_value(self, t2m):
        tz = xr.DataArray(
            data=[np.timedelta64(1, "h"), np.timedelta64(5, "h"), np.timedelta64(7, "h")],
            dims=["x"],
            coords={"x": t2m.coords["x"]},
            name="timezone",
        )
        dmt_array = excess_heat.daily_mean_temperature(t2m, day_start=0, time_shift=tz)
        dmt_coord = excess_heat.daily_mean_temperature(
            t2m.assign_coords({"timezone": tz}), day_start=0, time_shift="timezone"
        )
        np.testing.assert_allclose(dmt_array, dmt_coord)

    @pytest.mark.parametrize("shift", [-5, 0, 3])
    def test_scalar_value_is_applied_everywhere(self, t2m, shift):
        dmt_scalar = excess_heat.daily_mean_temperature(t2m, day_start=0, time_shift=shift)
        dmt_array = excess_heat.daily_mean_temperature(
            t2m.assign_coords({"timezone": ("x", 3 * [np.timedelta64(shift, "h")])}),
            day_start=0,
            time_shift="timezone",
        )
        np.testing.assert_allclose(dmt_scalar, dmt_array)

    @pytest.mark.parametrize("shift", [-2, -5, -9])
    def test_negative_value_means_day_starts_late(self, t2m, shift):
        dmt = excess_heat.daily_mean_temperature(t2m, day_start=0, time_shift=shift)
        ref = (np.asarray([11.5, 35.5, 59.5]) - shift)[:, None] * np.asarray([1e0, 1e1, 1e2])[None, :]
        np.testing.assert_allclose(dmt, ref)

    @pytest.mark.parametrize("shift", [2, 5, 9])
    def test_positive_value_means_day_starts_early(self, t2m, shift):
        dmt = excess_heat.daily_mean_temperature(t2m, day_start=0, time_shift=shift)
        ref = (np.asarray([35.5, 59.5, 83.5]) - shift)[:, None] * np.asarray([1e0, 1e1, 1e2])[None, :]
        np.testing.assert_allclose(dmt, ref)

    def test_multiple_timezones_grouping_of_duplicate_values(self, t2m):
        t2m = t2m.assign_coords(
            {"timezone": ("x", [np.timedelta64(1, "h"), np.timedelta64(4, "h"), np.timedelta64(4, "h")])}
        )
        dmt = excess_heat.daily_mean_temperature(t2m, day_start=0, time_shift="timezone")
        np.testing.assert_allclose(dmt, [[34.5, 315.0, 3150.0], [58.5, 555.0, 5550.0], [82.5, 795.0, 7950.0]])

    def test_multiple_timezones_fills_with_nan_to_preserve_outputs(self, t2m):
        t2m = t2m.assign_coords(
            {"timezone": ("x", [np.timedelta64(-2, "h"), np.timedelta64(0, "h"), np.timedelta64(10, "h")])}
        )
        dmt = excess_heat.daily_mean_temperature(t2m, day_start=0, time_shift="timezone")
        np.testing.assert_allclose(
            dmt,
            [
                [13.5, 115.0, np.nan],
                [37.5, 355.0, 2550.0],
                [61.5, 595.0, 4950.0],
                [np.nan, 835.0, 7350.0],
            ],
        )


def test_daily_mean_temperature_combined_day_start_and_time_shift_args():
    nx, nt = 4, 4 * 24
    t2m = xr.DataArray(
        data=np.arange(nt).repeat(nx).reshape((nt, nx)),
        dims=["valid_time", "x"],
        coords={
            "valid_time": pd.date_range("2026-01-01", periods=nt, freq="1h"),
            "x": np.arange(nx),
            "timezone": (
                "x",
                [
                    np.timedelta64(-5, "h"),
                    np.timedelta64(1, "h"),
                    np.timedelta64(3, "h"),
                    np.timedelta64(12, "h"),
                ],
            ),
        },
        name="t2m",
    )
    dmt = excess_heat.daily_mean_temperature(t2m, day_start=3, time_shift="timezone")
    np.testing.assert_allclose(
        dmt,
        [
            [19.5, 13.5, 11.5, np.nan],
            [43.5, 37.5, 35.5, 26.5],
            [67.5, 61.5, 59.5, 50.5],
            [np.nan, np.nan, 83.5, 74.5],
        ],
    )
