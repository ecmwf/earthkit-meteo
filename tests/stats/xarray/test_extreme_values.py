# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import numpy as np
import pytest

import earthkit.meteo.stats.xarray as stats

xr = pytest.importorskip("xarray")


@pytest.fixture
def da_2d():
    data = np.asarray(
        [
            [0.3, 3.0, 30.0, 300.0],
            [0.5, 5.0, 50.0, 500.0],
            [0.7, 7.0, 70.0, 700.0],
            [0.3, 3.0, 30.0, 300.0],
            [0.4, 4.0, 40.0, 400.0],
            [0.6, 6.0, 60.0, 600.0],
            [0.7, 7.0, 70.0, 700.0],
        ]
    )
    coords = {
        "foo": [2, 3, 4, 5, 6, 7, 8],
        "bar": [1, 5, 10, 20],
    }
    dims = ("foo", "bar")
    return xr.DataArray(data, coords, dims)


def test_fit_gumbel_over_foo_dim_detects_metadata(da_2d):
    gumbel = stats.fit_gumbel(da_2d, dim="foo")
    assert gumbel.ndim == 1
    assert gumbel.shape == (4,)
    assert gumbel.dims == ("bar",)
    assert len(gumbel.coords) == 1
    assert "bar" in gumbel.coords
    np.testing.assert_array_equal(gumbel.coords["bar"], da_2d.coords["bar"])


def test_fit_gumbel_fails_if_over_dimension_missing(da_2d):
    with pytest.raises(ValueError):
        stats.fit_gumbel(da_2d, dim="baz")


def test_return_period_to_value_fails_on_shared_dimension(da_2d):
    gumbel = stats.fit_gumbel(da_2d, dim="foo")
    rps = xr.DataArray([3, 5], coords={"bar": [-1, -2]}, dims=["bar"])
    with pytest.raises(ValueError):
        stats.return_period_to_value(rps, gumbel)


def test_fit_gumbel_over_bar_dim_detects_metadata(da_2d):
    gumbel = stats.fit_gumbel(da_2d, dim="bar")
    assert gumbel.ndim == 1
    assert gumbel.shape == (7,)
    assert gumbel.dims == ("foo",)
    assert len(gumbel.coords) == 1
    assert "foo" in gumbel.coords
    np.testing.assert_allclose(gumbel.coords["foo"], da_2d.coords["foo"])


def test_value_to_return_period_fails_on_shared_dimension(da_2d):
    gumbel = stats.fit_gumbel(da_2d, dim="foo")
    vals = xr.DataArray([1.0, 2.0, 3.0], coords={"bar": [4.0, 2.0, 0.0]}, dims=["bar"])
    with pytest.raises(ValueError):
        stats.value_to_return_period(vals, gumbel)
