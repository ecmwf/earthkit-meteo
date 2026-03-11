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

from earthkit.meteo.thermo import excess_heat
from earthkit.meteo.utils.testing import NO_TRANSFORMS
from earthkit.meteo.utils.testing import NO_XARRAY

pytestmark = pytest.mark.skipif(NO_XARRAY, reason="requires xarray")
pytestmark = pytest.mark.skipif(NO_TRANSFORMS, reason="requires earthkit.transforms")


class ProcessingPipelineMixin:

    @pytest.fixture
    def ehi_sig(self, dmt):
        return excess_heat.significance_index(dmt)

    @pytest.fixture
    def ehi_accl(self, dmt):
        return excess_heat.acclimatisation_index(dmt)

    @pytest.fixture
    def exhf(self, ehi_sig, ehi_accl):
        return excess_heat.excess_heat_factor(ehi_sig, ehi_accl)

    @pytest.fixture
    def excf(self, ehi_sig, ehi_accl):
        return excess_heat.excess_cold_factor(ehi_sig, ehi_accl)

    @pytest.fixture
    def hsev(self, exhf):
        return excess_heat.heatwave_severity(exhf)


class TestMetadata(ProcessingPipelineMixin):
    """Validate metadata generation for outputs"""

    @pytest.fixture
    def dmt(self):
        return xr.DataArray(
            [300.0, 290.0],
            coords={"valid_time": [np.datetime64("2025-01-01"), np.datetime64("2025-01-02")]},
            dims=["valid_time"],
            attrs={"units": "degC"},
        )

    def test_significance_index_output_metadata(self, ehi_sig):
        assert ehi_sig.name == "ehi_sig"
        assert ehi_sig.attrs["long_name"] == "Significance index"
        # assert ehi_sig.attrs["units"] == "degC"

    def test_acclimatisation_index_output_metadata(self, ehi_accl):
        assert ehi_accl.name == "ehi_accl"
        assert ehi_accl.attrs["long_name"] == "Acclimatisation index"
        # assert ehi_accl.attrs["units"] == "degC"

    def test_excess_heat_factor_output_metadata(self, exhf):
        assert exhf.name == "exhf"
        assert exhf.attrs["long_name"] == "Excess heat factor"
        assert exhf.attrs["units"] in {"K ** 2", "K^2", "K²"}

    def test_excess_heatwave_severity_metadata(self, hsev):
        assert hsev.name == "hsev"
        assert hsev.attrs["long_name"] == "Heatwave severity"
        assert "units" not in hsev.attrs or hsev.attrs["units"] == "1"

    def test_excess_cold_factor_metadata(self, excf):
        assert excf.name == "excf"
        assert excf.attrs["long_name"] == "Excess cold factor"
        assert excf.attrs["units"] in {"K ** 2", "K^2", "K²"}


class TestDailyMeanTemperatureArgDayStart:
    """Validate support for custom definition of day"""

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
    """Validate support for local timezones"""

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
        xr.testing.assert_allclose(dmt_tdelta, dmt_number)

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
            t2m.assign_coords({"timezone": ("x", t2m["x"].size * [np.timedelta64(shift, "h")])}),
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
        dmt_expected = [
            [13.5, 115.0, np.nan],
            [37.5, 355.0, 2550.0],
            [61.5, 595.0, 4950.0],
            [np.nan, 835.0, 7350.0],
        ]
        np.testing.assert_allclose(dmt, dmt_expected)


class TestDailyMeanTemperatureCombinedDayStartAndTimeShiftArgs:
    """Validate combined use of local timezone and custom definition of day"""

    nx = 4
    nt = 4 * 24

    @pytest.fixture
    def t2m(self):
        return xr.DataArray(
            data=np.arange(self.nt).repeat(self.nx).reshape((self.nt, self.nx)),
            dims=["valid_time", "x"],
            coords={
                "valid_time": pd.date_range("2026-01-01", periods=self.nt, freq="1h"),
                "x": np.arange(self.nx),
                "timezone": (
                    "x",
                    [
                        np.timedelta64(-5, "h"),  # net shift -8
                        np.timedelta64(1, "h"),  # net shift -2
                        np.timedelta64(3, "h"),  # net shift 0
                        np.timedelta64(12, "h"),  # net shift 9
                    ],
                ),
            },
            name="t2m",
        )

    def test_applied_net_shifts(self, t2m):
        dmt = excess_heat.daily_mean_temperature(t2m, day_start=3, time_shift="timezone")
        dmt_expected = [
            [19.5, 13.5, 11.5, np.nan],
            [43.5, 37.5, 35.5, 26.5],
            [67.5, 61.5, 59.5, 50.5],
            [np.nan, np.nan, 83.5, 74.5],
        ]
        np.testing.assert_allclose(dmt, dmt_expected)


# TODO
# day = np.linspace(0, 150, 151)
# # Approximate recreation of doi:10.3390/ijerph120100227, Fig. 4
# dmt = 25 + 10 * np.exp(-((day - 60) ** 2) / 5)
# time = np.datetime64("2025-01-01") + np.timedelta64(24, "h") * day
# return xr.DataArray(dmt, coords={"valid_time": time}, dims=["valid_time"], attrs={"units": "degC"})


def test_heatwave_severity_with_given_threshold():
    da = xr.DataArray([[0.0, 1.0, 3.0, 4.0], [1.0, 2.0, 1.0, 0.0]], dims=["foo", "time"])
    tr = xr.DataArray([2.5, 2.0], dims=["foo"])
    hsev = excess_heat.heatwave_severity(da, threshold=tr)
    xr.testing.assert_allclose(
        hsev, xr.DataArray([[0.0, 0.4, 1.2, 1.6], [0.5, 1.0, 0.5, 0.0]], dims=["foo", "time"])
    )
