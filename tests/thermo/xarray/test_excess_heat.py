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

pytest.importorskip("earthkit.transforms")
pytest.importorskip("thermofeel")
xr = pytest.importorskip("xarray")


def test_daily_mean_temperature_with_time_shift():
    nx = 4
    nt = 4 * 24

    t2m = xr.DataArray(
        data=np.arange(nt).repeat(nx).reshape((nt, nx)),
        dims=["valid_time", "x"],
        coords={
            "valid_time": pd.date_range("2026-01-01", periods=nt, freq="1h"),
            "x": np.arange(nx),
            "timezone": (
                "x",
                [
                    np.timedelta64(-8, "h"),
                    np.timedelta64(-2, "h"),
                    np.timedelta64(0, "h"),
                    np.timedelta64(9, "h"),
                ],
            ),
        },
        name="t2m",
    )

    dmt = excess_heat.daily_mean_temperature(t2m, time_shift="timezone", remove_partial_periods=True)
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

    def test_automatic_threshold(self, dmt):
        threshold = dmt.quantile(0.95).drop_vars("quantile")
        xr.testing.assert_equal(
            excess_heat.significance_index(dmt),
            excess_heat.significance_index(dmt, threshold=threshold),
        )

    @pytest.mark.parametrize("quantile", [0.05, 0.4, 0.7])
    def test_threshold_quantile(self, dmt, quantile):
        threshold = dmt.quantile(quantile).drop_vars("quantile")
        xr.testing.assert_equal(
            excess_heat.significance_index(dmt, threshold_q=quantile),
            excess_heat.significance_index(dmt, threshold=threshold),
        )

    # TODO: test with dayofyear climatology


class TestAcclimatisationIndex:
    @pytest.fixture
    def dmt(self):
        values = [0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        time = pd.date_range("2025-01-01", periods=len(values), freq="1d")
        return xr.DataArray(values, dims=["time"], coords={"time": time})

    def test_metadata(self, dmt):
        ehi_accl = excess_heat.acclimatisation_index(dmt, ndays_ref=10)
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
            excess_heat.heatwave_severity(exhf, threshold_q=quantile),
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
    """Validation with the short heatwave case of Nairn (2015), Fig. 2."""

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
