# (C) Copyright 2025 - ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import numpy as np
import pandas as pd
import pytest

from earthkit.meteo.thermo import excess_heat
from earthkit.meteo.utils.testing import NO_TRANSFORMS

pytestmark = pytest.mark.skipif(NO_TRANSFORMS, reason="requires earthkit.transforms")

xr = pytest.importorskip("xarray")


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

    def test_net_shifts_by_result_values(self, t2m):
        dmt = excess_heat.daily_mean_temperature(t2m, day_start=3, time_shift="timezone")
        dmt_expected = [
            [19.5, 13.5, 11.5, np.nan],
            [43.5, 37.5, 35.5, 26.5],
            [67.5, 61.5, 59.5, 50.5],
            [np.nan, np.nan, 83.5, 74.5],
        ]
        np.testing.assert_allclose(dmt, dmt_expected)


class TestSignificanceIndex:

    @pytest.fixture
    def dmt(self):
        values = [0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        time = pd.date_range("2025-01-01", periods=len(values), freq="1d")
        return xr.DataArray(values, dims=["time"], coords={"time": time})

    def test_metadata(self, dmt):
        ehi_sig = excess_heat.significance_index(dmt)
        assert ehi_sig.name == "ehi_sig"
        assert ehi_sig.attrs["long_name"] == "Significance index"
        # assert ehi_sig.attrs["units"] == "degC"

    def test_identity_edge_case(self, dmt):
        ehi_sig = excess_heat.significance_index(dmt, ndays=1, threshold=0.0)
        xr.testing.assert_allclose(ehi_sig, dmt)

    def test_ndays_3(self, dmt):
        ehi_sig = excess_heat.significance_index(dmt, ndays=3, threshold=5.0)
        ref = dmt.isel({"time": slice(None, -2)}).copy(
            data=[-5.0, -5.0, -5.0, -2.0, 1.0, 4.0, 4.0, 4.0, 1.0, -2.0, -5.0, -5.0, -5.0]
        )
        xr.testing.assert_allclose(ehi_sig, ref)

    def test_ndays_5(self, dmt):
        ehi_sig = excess_heat.significance_index(dmt, ndays=5, threshold=4.0)
        ref = dmt.isel({"time": slice(None, -4)}).copy(
            data=[-4.0, -2.2, -0.4, 1.4, 3.2, 5.0, 3.2, 1.4, -0.4, -2.2, -4.0]
        )
        xr.testing.assert_allclose(ehi_sig, ref)

    @pytest.mark.parametrize("quantile", [0.05, 0.4, 0.7])
    def test_threshold_quantile(self, dmt, quantile):
        threshold = dmt.quantile(quantile).drop_vars("quantile")
        xr.testing.assert_equal(
            excess_heat.significance_index(dmt, threshold_quantile=quantile),
            excess_heat.significance_index(dmt, threshold=threshold),
        )

    @pytest.mark.parametrize("start", ["2025-01-04", "2025-01-08", "2025-01-10"])
    def test_threshold_period(self, dmt, start):
        q = 0.5
        period = slice(start, None)
        threshold = dmt.sel(time=period).quantile(q).drop_vars("quantile")
        xr.testing.assert_equal(
            excess_heat.significance_index(dmt, threshold_period=period, threshold_quantile=q),
            excess_heat.significance_index(dmt, threshold=threshold),
        )


class TestAcclimatisationIndex:

    @pytest.fixture
    def dmt(self):
        values = [0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        time = pd.date_range("2025-01-01", periods=len(values), freq="1d")
        return xr.DataArray(values, dims=["time"], coords={"time": time})

    def test_metadata(self, dmt):
        ehi_accl = excess_heat.acclimatisation_index(dmt)
        assert ehi_accl.name == "ehi_accl"
        assert ehi_accl.attrs["long_name"] == "Acclimatisation index"
        # assert ehi_accl.attrs["units"] == "degC"

    def test_nday_ref_1_ndays_1(self, dmt):
        ehi_accl = excess_heat.acclimatisation_index(dmt, ndays_ref=1, ndays=1)
        ref = dmt.isel({"time": slice(1, None)}).copy(
            data=[0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, -9.0, 0.0, 0.0, 0.0, 0.0]
        )
        xr.testing.assert_allclose(ehi_accl, ref)

    def test_nday_ref_5_ndays_3(self, dmt):
        ehi_accl = excess_heat.acclimatisation_index(dmt, ndays_ref=5, ndays=3)
        ref = dmt.isel({"time": slice(5, -2)}).copy(data=[9.0, 7.2, 5.4, 0.6, -4.2, -9.0, -7.2, -5.4])
        xr.testing.assert_allclose(ehi_accl, ref)


class TestExcessHeatFactor:

    @pytest.fixture
    def ehi_sig(self):
        return xr.DataArray([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0])

    @pytest.fixture
    def ehi_accl(self):
        return xr.DataArray([3.0, 0.0, -4.0, 3.0, 0.0, -4.0, 3.0, 0.0, -4.0])

    def test_metadata(self, ehi_sig, ehi_accl):
        exhf = excess_heat.excess_heat_factor(ehi_sig, ehi_accl)
        assert exhf.name == "exhf"
        assert exhf.attrs["long_name"] == "Excess heat factor"
        assert exhf.attrs["units"] in {"K ** 2", "K^2", "K²"}

    def test_with_clip_false(self, ehi_sig, ehi_accl):
        exhf = excess_heat.excess_heat_factor(ehi_sig, ehi_accl, clip=False)
        np.testing.assert_allclose(exhf, [15.0, 5.0, 5.0, 0.0, 0.0, 0.0, -6.0, -2.0, -2.0])

    def test_with_clip_true(self, ehi_sig, ehi_accl):
        exhf = excess_heat.excess_heat_factor(ehi_sig, ehi_accl, clip=True)
        np.testing.assert_allclose(exhf, [15.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class TestHeatwaveSeverity:

    @pytest.fixture
    def exhf(self):
        return xr.DataArray([[0.0, 1.0, 3.0, 4.0], [1.0, 2.0, 1.0, 0.0]], dims=["foo", "time"])

    def test_defaults_and_metadata(self, exhf):
        hsev = excess_heat.heatwave_severity(exhf)
        assert hsev.name == "hsev"
        assert hsev.attrs["long_name"] == "Heatwave severity"
        assert "units" not in hsev.attrs or hsev.attrs["units"] == "1"

    def test_heatwave_severity_with_fixed_threshold(self, exhf):
        tr = xr.DataArray([2.5, 2.0], dims=["foo"])
        hsev = excess_heat.heatwave_severity(exhf, threshold=tr)
        ref = xr.DataArray([[0.0, 0.4, 1.2, 1.6], [0.5, 1.0, 0.5, 0.0]], dims=["foo", "time"])
        xr.testing.assert_allclose(hsev, ref)

    @pytest.mark.parametrize("quantile", [0.05, 0.4, 0.7])
    def test_threshold_quantile(self, exhf, quantile):
        threshold = exhf.where(exhf > 0).quantile(quantile, dim="time").drop_vars("quantile")
        xr.testing.assert_equal(
            excess_heat.heatwave_severity(exhf, threshold_quantile=quantile),
            excess_heat.heatwave_severity(exhf, threshold=threshold),
        )


class TestExcessColdFactor:

    @pytest.fixture
    def ehi_sig(self):
        return xr.DataArray([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0])

    @pytest.fixture
    def ehi_accl(self):
        return xr.DataArray([3.0, 0.0, -4.0, 3.0, 0.0, -4.0, 3.0, 0.0, -4.0])

    def test_excess_cold_factor_metadata(self, ehi_sig, ehi_accl):
        excf = excess_heat.excess_cold_factor(ehi_sig, ehi_accl)
        assert excf.name == "excf"
        assert excf.attrs["long_name"] == "Excess cold factor"
        assert excf.attrs["units"] in {"K ** 2", "K^2", "K²"}

    def test_with_clip_false(self, ehi_sig, ehi_accl):
        exhf = excess_heat.excess_cold_factor(ehi_sig, ehi_accl, clip=False)
        np.testing.assert_allclose(exhf, [5.0, 5.0, 20.0, 0.0, 0.0, 0.0, -2.0, -2.0, -8.0])

    def test_with_clip_true(self, ehi_sig, ehi_accl):
        exhf = excess_heat.excess_cold_factor(ehi_sig, ehi_accl, clip=True)
        np.testing.assert_allclose(exhf, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -8.0])


class TestWithShortHeatwaveSyntheticData:
    """Validation with the short heatwave case of Nairn (2015), Fig. 2"""

    @pytest.fixture
    def dmt(self):
        day = np.linspace(0, 150, 151)
        dmt = 25 + 10 * np.exp(-((day - 68) ** 2) / 5)
        time = np.datetime64("2025-01-01") + np.timedelta64(24, "h") * day
        return xr.DataArray(dmt, coords={"valid_time": time}, dims=["valid_time"], attrs={"units": "degC"})

    @pytest.fixture
    def ehi_sig(self, dmt):
        return excess_heat.significance_index(dmt, threshold=30.0, ndays=3).compute()

    @pytest.fixture
    def ehi_accl(self, dmt):
        return excess_heat.acclimatisation_index(dmt, ndays_ref=30, ndays=3).compute()

    @pytest.fixture
    def exhf(self, ehi_sig, ehi_accl):
        return excess_heat.excess_heat_factor(ehi_sig, ehi_accl)

    def test_dmt_expectations(self, dmt):
        assert dmt.idxmax() == np.datetime64("2025-03-10")
        imax = np.argmax(dmt.values)
        np.testing.assert_equal(np.where(dmt > 30)[0], [imax - 1, imax, imax + 1])

    def test_significance_index_properties(self, ehi_sig):
        assert ehi_sig.idxmax() == np.datetime64("2025-03-09")
        np.testing.assert_equal(
            ehi_sig["valid_time"][ehi_sig > 0].values,
            [np.datetime64("2025-03-08"), np.datetime64("2025-03-09"), np.datetime64("2025-03-10")],
        )

    def test_acclimatisation_index_properties(self, ehi_accl):
        assert ehi_accl.idxmax() == np.datetime64("2025-03-09")
        np.testing.assert_array_less(ehi_accl.sel({"valid_time": slice("2025-03-13", None)}), 0.001)
        assert ehi_accl["valid_time"].min() == np.datetime64("2025-01-31")

    def test_excess_heat_factor_properties(self, exhf):
        assert exhf.idxmax() == np.datetime64("2025-03-09")
        assert exhf.idxmin() == np.datetime64("2025-03-06")
        assert 30 < exhf.max().item() < 35
        assert exhf.sel(valid_time="2025-03-08") > exhf.sel({"valid_time": "2025-03-10"})
