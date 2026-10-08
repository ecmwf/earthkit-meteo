# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import pytest
from _vertical_cases import *  # noqa: F401,F403

import earthkit.meteo.vertical as vertical_high_level


@pytest.fixture
def vertical():
    return vertical_high_level


@pytest.mark.parametrize("input_type", ["numpy", "xarray", "fieldlist"])
def test_high_level_extrapolate_below_surface_dispatch(input_type):
    import numpy as np

    t_sfc = np.array([288.15, 290.0, 275.0, 290.0])
    h_sfc = np.array([100.0, 2200.0, 3000.0, 3000.0])
    p_sfc = np.array([100000.0, 78000.0, 70000.0, 70000.0])
    ref = [290.83740533454284, 303.07481159412237, 297.05478853917333, 299.32528337706367]

    if input_type == "numpy":
        expected_type = np.ndarray
        args = (t_sfc, h_sfc, p_sfc)
    elif input_type == "xarray":
        xr = pytest.importorskip("xarray")
        expected_type = xr.DataArray
        args = tuple(xr.DataArray(v, dims="cell") for v in (t_sfc, h_sfc, p_sfc))
    else:
        from earthkit.meteo.utils.testing import NO_EKD

        if NO_EKD:
            pytest.skip("EKD is not installed")
        from earthkit.data import Field, FieldList

        expected_type = FieldList
        args = tuple(
            FieldList.from_fields([Field.from_components(values=v, vertical={"level": 0, "level_type": "surface"})])
            for v in (t_sfc, h_sfc, p_sfc)
        )

    res = vertical_high_level.extrapolate_temperature_below_surface(*args, 105000.0)

    assert isinstance(res, expected_type)
    values = res.to_numpy() if input_type == "fieldlist" else np.asarray(res)
    np.testing.assert_allclose(np.ravel(values), ref, rtol=1e-10)
