# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np

from earthkit.meteo.vertical.array import extrapolate_geopotential_below_surface
from earthkit.meteo.vertical.array import extrapolate_temperature_below_surface

# The surface points cover the regimes of the temperature lapse rate:
#   - h < 2000 m: standard lapse rate
#   - 2000 m <= h <= 2500 m: blended T0', capped at 298 K
#   - h > 2500 m: T0' = T0 (below 298 K)
#   - h > 2500 m: T0' capped at 298 K
T_SFC = np.array([288.15, 290.0, 275.0, 290.0])
H_SFC = np.array([100.0, 2200.0, 3000.0, 3000.0])
P_SFC = np.array([100000.0, 78000.0, 70000.0, 70000.0])
TARGET_P = np.array([85000.0, 105000.0])

REF_T = np.array(
    [
        [279.3761404060049, 293.7211819249435, 285.3488956941758, 294.42857704742926],
        [290.83740533454284, 303.07481159412237, 297.05478853917333, 299.32528337706367],
    ]
)
REF_Z = np.array(
    [
        [14217.921951915694, 14361.339585332338, 13806.411852417126, 12954.764317094425],
        [-3073.864772922168, -3883.7196996812127, -3854.448773489843, -5669.41597931656],
    ]
)


def test_array_extrapolate_temperature_below_surface():
    res = extrapolate_temperature_below_surface(T_SFC, H_SFC, P_SFC, TARGET_P)

    assert res.shape == (2, 4)
    np.testing.assert_allclose(res, REF_T, rtol=1e-10)


def test_array_extrapolate_geopotential_below_surface():
    res = extrapolate_geopotential_below_surface(T_SFC, H_SFC, P_SFC, TARGET_P)

    assert res.shape == (2, 4)
    np.testing.assert_allclose(res, REF_Z, rtol=1e-10)
