# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import numpy as np
import pytest

from earthkit.meteo import stats
from earthkit.meteo.utils.testing import NO_XARRAY

pytestmark = pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")


@pytest.fixture
def da_2d():
    import xarray as xr

    data = np.asarray(
        [
            [0.3, 3.0, 30.0],
            [0.5, 5.0, 50.0],
            [0.7, 7.0, 70.0],
            [0.3, 3.0, 30.0],
            [0.4, 4.0, 40.0],
            [0.6, 6.0, 60.0],
            [0.7, 7.0, 70.0],
        ]
    )
    coords = {
        "foo": [2, 3, 4, 5, 6, 7, 8],
        "bar": [1, 5, 10],
    }
    dims = ("foo", "bar")
    return xr.DataArray(data, coords, dims)


def test_fit_gumbel_over_foo_dim_detects_metadata(da_2d):
    gumbel = stats.fit_gumbel(da_2d, over="foo")
    assert gumbel.ndim == 1
    assert gumbel.shape == (3,)
    assert gumbel.dims == ("bar",)
    assert len(gumbel.coords) == 1
    assert "bar" in gumbel.coords
    np.testing.assert_array_equal(gumbel.coords["bar"], da_2d.coords["bar"])


def test_fit_gumbel_fails_if_over_dimension_missing(da_2d):
    with pytest.raises(ValueError):
        stats.fit_gumbel(da_2d, over="baz")


def test_return_period_to_value(da_2d):
    import xarray as xr

    gumbel = stats.fit_gumbel(da_2d, over="foo")
    rps = xr.DataArray([3, 5], coords={"baz": [-1, -2]}, dims=["baz"])
    result = stats.return_period_to_value(rps, gumbel)
    assert result.shape == (2, 3)
    assert result.dims == ("baz", "bar")
    np.testing.assert_array_equal(result.coords["bar"], da_2d.coords["bar"])
    np.testing.assert_array_equal(result.coords["baz"], rps.coords["baz"])
    # Ensure consistency between array and xarray implementations
    result_arr = stats.return_period_to_value(rps.values, gumbel)
    np.testing.assert_array_equal(result, result_arr)


def test_return_period_to_value_fails_on_shared_dimension(da_2d):
    import xarray as xr

    gumbel = stats.fit_gumbel(da_2d, over="foo")
    rps = xr.DataArray([3, 5], coords={"bar": [-1, -2]}, dims=["bar"])
    with pytest.raises(ValueError):
        stats.return_period_to_value(rps, gumbel)


def test_fit_gumbel_over_bar_dim_detects_metadata(da_2d):
    gumbel = stats.fit_gumbel(da_2d, over="bar")
    assert gumbel.ndim == 1
    assert gumbel.shape == (7,)
    assert gumbel.dims == ("foo",)
    assert len(gumbel.coords) == 1
    assert "foo" in gumbel.coords
    np.testing.assert_allclose(gumbel.coords["foo"], da_2d.coords["foo"])


def test_value_to_return_period(da_2d):
    import xarray as xr

    gumbel = stats.fit_gumbel(da_2d, over="bar")
    vals = xr.DataArray([1.0, 2.0, 3.0], coords={"baz": [4.0, 2.0, 0.0]}, dims=["baz"])
    result = stats.value_to_return_period(vals, gumbel)
    assert result.shape == (3, 7)
    assert result.dims == ("baz", "foo")
    np.testing.assert_array_equal(result.coords["foo"], da_2d.coords["foo"])
    np.testing.assert_array_equal(result.coords["baz"], vals.coords["baz"])
    # Ensure consistency between array and xarray implementations
    result_arr = stats.value_to_return_period(vals.values, gumbel)
    np.testing.assert_array_equal(result, result_arr)


def test_value_to_return_period_fails_on_shared_dimension(da_2d):
    import xarray as xr

    gumbel = stats.fit_gumbel(da_2d, over="foo")
    vals = xr.DataArray([1.0, 2.0, 3.0], coords={"bar": [4.0, 2.0, 0.0]}, dims=["bar"])
    with pytest.raises(ValueError):
        stats.value_to_return_period(vals, gumbel)
