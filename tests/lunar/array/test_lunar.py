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
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo.lunar.array import lunar

# Test case: full moon on 5 April 2004 at 12:03 UTC
# Sub-lunar point:  lat=5.15, lon=164.2

# Currently only numpy is supported, but we keep the parametrization for future support of other array libraries.
_NAMESPACE_DEVICES = list(filter(lambda x: x[0]._earthkit_array_namespace_name == "numpy", NAMESPACE_DEVICES))


@pytest.mark.parametrize(
    "date,v_ref",
    [
        (datetime.datetime(2004, 4, 5, 12, 3, 0), 369020.9437902331),
    ],
)
def test_distance_from_earth_centre_to_moon(date, v_ref):
    v = lunar.distance_from_earth_centre_to_moon(date)
    assert np.allclose(v, v_ref)


@pytest.mark.parametrize("xp, device", _NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "date,lat,lon,v_ref",
    [
        (datetime.datetime(2004, 4, 5, 12, 3, 0), -5, 164, 362933.3708641256),
        (datetime.datetime(2004, 4, 5, 12, 3, 0), [-5, 5], [164, -16], [362933.3708641256, 375118.1672196273]),
    ],
)
def test_distance_to_moon(xp, device, date, lat, lon, v_ref):
    lat = xp.asarray(lat, device=device)
    lon = xp.asarray(lon, device=device)
    v_ref = xp.asarray(v_ref, device=device)
    v = lunar.distance_to_moon(date, lat, lon)
    v_ref = xp.asarray(v_ref, dtype=v.dtype)
    assert xp.allclose(v, v_ref)


@pytest.mark.parametrize("xp, device", _NAMESPACE_DEVICES)
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
def test_delta_distance_to_moon(xp, device, date, lat, lon, v_ref):
    lat = xp.asarray(lat, device=device)
    lon = xp.asarray(lon, device=device)
    v_ref = xp.asarray(v_ref, device=device)
    v = lunar.delta_distance_to_moon(date, lat, lon)
    v_ref = xp.asarray(v_ref, dtype=v.dtype)
    assert xp.allclose(v, v_ref)
