# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest

from earthkit.meteo.utils.testing import NO_XARRAY

pytestmark = pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")

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


def _da(values):
    import xarray as xr

    return xr.DataArray(values, dims="cell", coords={"cell": np.arange(len(values))})


def _levels():
    import xarray as xr

    return xr.DataArray(TARGET_P, dims="z", coords={"z": TARGET_P})


def test_xr_extrapolate_temperature_below_surface():
    from earthkit.meteo.vertical.xarray import extrapolate_temperature_below_surface

    res = extrapolate_temperature_below_surface(_da(T_SFC), _da(H_SFC), _da(P_SFC), _levels())

    assert set(res.dims) == {"cell", "z"}
    np.testing.assert_allclose(res["z"].values, TARGET_P)
    assert res.attrs["standard_name"] == "air_temperature"
    assert res.attrs["units"] == "K"
    np.testing.assert_allclose(res.transpose("z", "cell").values, REF_T, rtol=1e-10)


def test_xr_extrapolate_geopotential_below_surface():
    from earthkit.meteo.vertical.xarray import extrapolate_geopotential_below_surface

    res = extrapolate_geopotential_below_surface(_da(T_SFC), _da(H_SFC), _da(P_SFC), _levels())

    assert set(res.dims) == {"cell", "z"}
    np.testing.assert_allclose(res["z"].values, TARGET_P)
    assert res.attrs["standard_name"] == "geopotential"
    assert res.attrs["units"] == "m2 s-2"
    np.testing.assert_allclose(res.transpose("z", "cell").values, REF_Z, rtol=1e-10)