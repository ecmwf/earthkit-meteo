# (C) Copyright 2025 - ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import numpy as np
import pandas as pd
import pytest

from earthkit.meteo.extreme import excess_heat


class TestDailyMeanTemeratureArgDayStart:

    @pytest.fixture
    def t2m(self):
        return pd.Series(
            data=np.arange(96),
            index=pd.period_range("2026-01-01", periods=96, freq="1h", name="valid_time"),
            name="t2m",
        ).to_xarray()

    def test_number_as_hours(self, t2m):
        dmt_tdelta = excess_heat.daily_mean_temperature(t2m, day_start=np.timedelta64(5, "h"))
        dmt_number = excess_heat.daily_mean_temperature(t2m, day_start=5)
        np.testing.assert_allclose(dmt_tdelta, dmt_number)

    def test_zero_start(self, t2m):
        dmt = excess_heat.daily_mean_temperature(t2m, day_start=0, time_shift=0)
        # 1st element: expect mean of [0, 1, ..., 23] = 11.5
        # 2nd element: expect mean of [24, 25, ..., 47] = 35.5
        # etc.
        np.testing.assert_allclose(dmt, [11.5, 35.5, 59.5, 83.5])

    def test_negative_start(self, t2m):
        # Negative start = day starts early, here: 22:00
        dmt = excess_heat.daily_mean_temperature(t2m, day_start=-2, time_shift=0)
        # Remove partial periods by default, only 3 elements returned
        # 1st element: expect mean of [22, 23, ..., 45] = 33.5
        # etc.
        np.testing.assert_allclose(dmt, [33.5, 57.5, 81.5])

    def test_positive_start(self, t2m):
        # Positive start = day starts late, here: 09:00
        dmt = excess_heat.daily_mean_temperature(t2m, day_start=9, time_shift=0)
        # Remove partial periods by default, only 3 elements returned
        # 1st element: expect mean of [8, 9, ..., 31] = 33.5
        # etc.
        np.testing.assert_allclose(dmt, [20.5, 44.5, 68.5])
