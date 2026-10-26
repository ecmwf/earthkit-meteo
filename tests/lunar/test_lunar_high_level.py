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

from earthkit.meteo import lunar

# Currently these are the same tests as in the array implementation, but we keep them separate for future
# support of other array libraries.


@pytest.mark.parametrize(
    "date,v_ref",
    [
        (datetime.datetime(2004, 4, 5, 12, 3, 0), 369020.9437902331),
    ],
)
def test_distance_from_earth_centre_to_moon_high(date, v_ref):
    v = lunar.distance_from_earth_centre_to_moon(date)
    assert np.allclose(v, v_ref)


@pytest.mark.parametrize(
    "date,lat,lon,v_ref",
    [
        (datetime.datetime(2004, 4, 5, 12, 3, 0), -5, 164, 362933.3708641256),
        (datetime.datetime(2004, 4, 5, 12, 3, 0), [-5, 5], [164, -16], [362933.3708641256, 375118.1672196273]),
    ],
)
def test_distance_to_moon_high(date, lat, lon, v_ref):
    xp = np
    device = None

    lat = xp.asarray(lat, device=device)
    lon = xp.asarray(lon, device=device)
    v_ref = xp.asarray(v_ref, device=device)
    v = lunar.distance_to_moon(date, lat, lon)
    v_ref = xp.asarray(v_ref, dtype=v.dtype)
    assert xp.allclose(v, v_ref)


@pytest.mark.parametrize(
    "date,lat,lon,v_ref",
    [
        (
            datetime.datetime(2004, 4, 5, 12, 3),
            [-5, 5, 70, 90],
            [164, -16, -16, -16],
            [0.0000000000, 12184.7963555017, 8684.1519151750, 6640.0936603358],
        ),
    ],
)
def test_delta_distance_to_moon_high(date, lat, lon, v_ref):
    xp = np
    device = None

    lat = xp.asarray(lat, device=device)
    lon = xp.asarray(lon, device=device)
    v_ref = xp.asarray(v_ref, device=device)
    v = lunar.delta_distance_to_moon(date, lat, lon)
    v_ref = xp.asarray(v_ref, dtype=v.dtype)
    assert xp.allclose(v, v_ref)
