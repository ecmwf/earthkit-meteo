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

import earthkit.meteo.vertical.array as array_vertical
import earthkit.meteo.vertical.xarray as vertical
from earthkit.meteo.utils.testing import NO_XARRAY

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})

pytestmark = pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")


def _da(values):
    import xarray as xr

    return xr.DataArray(np.asarray(values))


@pytest.mark.parametrize("z", [[1000.0, 10000.0]])
def test_xr_geopotential_height_from_geopotential(z):
    out = vertical.geopotential_height_from_geopotential(_da(z))
    ref = array_vertical.geopotential_height_from_geopotential(np.asarray(z))
    assert np.allclose(out.values, ref)
    assert out.name == "gh"
    assert out.attrs.get("units") == "m"
    assert out.attrs.get("standard_name") == "geopotential_height"


@pytest.mark.parametrize("gh", [[100.0, 1000.0]])
def test_xr_geopotential_from_geopotential_height(gh):
    out = vertical.geopotential_from_geopotential_height(_da(gh))
    ref = array_vertical.geopotential_from_geopotential_height(np.asarray(gh))
    assert np.allclose(out.values, ref)
    assert out.name == "z"
    assert out.attrs.get("units") == "m2 s-2"
    assert out.attrs.get("standard_name") == "geopotential"


@pytest.mark.parametrize("h", [[100.0, 5000.0]])
def test_xr_geopotential_height_from_geometric_height(h):
    out = vertical.geopotential_height_from_geometric_height(_da(h))
    ref = array_vertical.geopotential_height_from_geometric_height(np.asarray(h))
    assert np.allclose(out.values, ref)
    assert out.name == "gh"
    assert out.attrs.get("units") == "m"
    assert out.attrs.get("standard_name") == "geopotential_height"


@pytest.mark.parametrize("h", [[100.0, 5000.0]])
def test_xr_geopotential_from_geometric_height(h):
    out = vertical.geopotential_from_geometric_height(_da(h))
    ref = array_vertical.geopotential_from_geometric_height(np.asarray(h))
    assert np.allclose(out.values, ref)
    assert out.name == "z"
    assert out.attrs.get("units") == "m2 s-2"
    assert out.attrs.get("standard_name") == "geopotential"


@pytest.mark.parametrize("gh", [[100.0, 5000.0]])
def test_xr_geometric_height_from_geopotential_height(gh):
    out = vertical.geometric_height_from_geopotential_height(_da(gh))
    ref = array_vertical.geometric_height_from_geopotential_height(np.asarray(gh))
    assert np.allclose(out.values, ref)
    assert out.name == "h"
    assert out.attrs.get("units") == "m"
    assert out.attrs.get("standard_name") == "altitude"


@pytest.mark.parametrize("z", [[1000.0, 10000.0]])
def test_xr_geometric_height_from_geopotential(z):
    out = vertical.geometric_height_from_geopotential(_da(z))
    ref = array_vertical.geometric_height_from_geopotential(np.asarray(z))
    assert np.allclose(out.values, ref)
    assert out.name == "h"
    assert out.attrs.get("units") == "m"
    assert out.attrs.get("standard_name") == "altitude"
