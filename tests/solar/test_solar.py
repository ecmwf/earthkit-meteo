# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime

import numpy as np
import pytest

from earthkit.meteo import solar


@pytest.mark.parametrize(
    "date,expected_value",
    [
        (datetime.datetime(2024, 4, 22), 112.0),
        (datetime.datetime(2024, 4, 22, 12, 0, 0), 112.5),
    ],
)
def test_julian_day(date, expected_value):
    v = solar.julian_day(date)
    assert np.isclose(v, expected_value)
