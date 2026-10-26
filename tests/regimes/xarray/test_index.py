# (C) Copyright 2025- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from earthkit.meteo.regimes import Patterns
from earthkit.meteo.regimes.xarray import project, regime_index


@pytest.fixture
def weights1d():
    return xr.DataArray(data=[1.0, 3.0], coords={"lat": (["lat"], [60.0, 50.0])}, dims=["lat"])


@pytest.fixture
def data3d():
    data2d = xr.DataArray(
        data=[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
        coords={
            "lat": (["lat"], [60.0, 50.0]),
            "lon": (["lon"], [-10.0, 0.0, 10.0, 20.0]),
        },
        dims=["lat", "lon"],
    )
    return data2d.expand_dims({"foo": [1.0, 2.0, 4.0]})


@pytest.fixture
def patterns():
    class MockPatterns(Patterns):
        def __init__(self):
            grid = {"grid": [1.0, 1.0], "area": [46.0, 0.0, 45.0, 3.0]}
            super().__init__(["a", "b"], grid=grid, xp=np)

        def patterns(self, **kwargs):
            self.received_kwargs = kwargs  # to verify mapped coordinates
            out = np.asarray([
                [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            ])
            if "foo" in kwargs:
                return kwargs["foo"][..., None, None, None] * out
            return out

    return MockPatterns()


def test_project_with_with_lower_dimensional_weights(data3d, patterns, weights1d):
    result = project(data3d, patterns, weights1d, patterns_coords=["foo"])
    assert result.dims == ("foo", "pattern")
    assert result.shape == (3, len(patterns))
    assert result.coords["pattern"].values.tolist() == ["a", "b"]
    # E.g.: first element (pattern="a", foo=1.)
    #     data             weights (normalised)         pattern
    # / 0  1  2  3 \ * / 1/16  1/16  1/16  1/16 \ * / 1  1  1  1 \  = /     0  1/16  2/16  3/16 \
    # \ 4  5  6  7 /   \ 3/16  3/16  3/16  3/16 /   \ 1  0  0  0 /    \ 12/16     0     0     0 /
    # sum of elements = 18/16 = 1.125
    np.testing.assert_allclose(result, [[1.125, 4.3125], [2.25, 8.625], [4.5, 17.25]])


def test_project_with_full_dimensional_weights(data3d, patterns):
    weights2d = data3d.sel(foo=1.0)
    result = project(data3d, patterns, weights2d, patterns_coords=["foo"])
    assert result.dims == ("foo", "pattern")
    assert result.shape == (3, len(patterns))
    assert result.coords["pattern"].values.tolist() == ["a", "b"]
    np.testing.assert_allclose(
        result,
        [
            [30.0 / 28.0, 135.0 / 28.0],
            [60.0 / 28.0, 270.0 / 28.0],
            [120.0 / 28.0, 540.0 / 28.0],
        ],
    )


def test_project_ensures_weight_dims_are_pattern_dims(data3d, patterns):
    with pytest.raises(ValueError):
        project(data3d, patterns, weights=data3d)  # dimension foo is not a pattern dim


def test_project_without_patterns_coords(data3d, patterns, weights1d):
    project(data3d, patterns, weights1d)
    assert not patterns.received_kwargs


def test_project_with_patterns_coords_none(data3d, patterns, weights1d):
    project(data3d, patterns, weights1d, patterns_coords=None)
    assert not patterns.received_kwargs


def test_project_with_patterns_coords_empty_mapping(data3d, patterns, weights1d):
    project(data3d, patterns, weights1d, patterns_coords={})
    assert not patterns.received_kwargs


def test_project_with_patterns_coords_empty_sequence(data3d, patterns, weights1d):
    project(data3d, patterns, weights1d, patterns_coords=[])
    assert not patterns.received_kwargs


def test_project_with_patterns_coords_mapping(data3d, patterns, weights1d):
    data4d = data3d.expand_dims({"baz": [4.0, 5.0]})
    result = project(data4d, patterns, weights1d, patterns_coords={"foo": "foo", "bar": "baz"})
    assert "foo" in patterns.received_kwargs
    np.testing.assert_equal(patterns.received_kwargs["foo"], [[1.0, 2.0, 4.0], [1.0, 2.0, 4.0]])
    assert "bar" in patterns.received_kwargs
    assert "bar" not in result.coords
    np.testing.assert_equal(patterns.received_kwargs["bar"], [[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])
    assert "baz" not in patterns.received_kwargs
    assert "baz" in result.coords


def test_project_with_patterns_coords_sequence(data3d, patterns, weights1d):
    data4d = data3d.expand_dims({"baz": [4.0, 5.0], "bar": [1.0]})
    result = project(data4d, patterns, weights1d, patterns_coords=["foo", "bar"])
    assert "foo" in patterns.received_kwargs
    np.testing.assert_equal(patterns.received_kwargs["foo"], [[1.0, 2.0, 4.0]])
    assert "bar" in patterns.received_kwargs
    assert "bar" in result.coords
    np.testing.assert_equal(patterns.received_kwargs["bar"], [[1.0, 1.0, 1.0]])
    assert "baz" not in patterns.received_kwargs
    assert "baz" in result.coords


def test_project_maintains_additional_dimensions_not_in_patterns_before(data3d, patterns, weights1d):
    data4d = data3d.expand_dims({"bar": [4.0, 5.0]})
    result = project(data4d, patterns, weights1d, patterns_coords=["foo"])
    assert result.dims == ("bar", "foo", "pattern")
    assert result.shape == (2, 3, 2)


def test_project_maintains_additional_dimensions_not_in_patterns_after(data3d, patterns, weights1d):
    data4d = data3d.expand_dims({"bar": [4.0, 5.0]}, axis=1)
    result = project(data4d, patterns, weights1d, patterns_coords=["foo"])
    assert result.dims == ("foo", "bar", "pattern")
    assert result.shape == (3, 2, 2)


def test_project_fails_when_trailing_shape_does_not_match_pattern_shape(data3d, patterns, weights1d):
    with pytest.raises(ValueError):
        project(data3d.transpose("foo", "lon", "lat"), patterns, weights1d, patterns_coords=["foo"])
    with pytest.raises(ValueError):
        project(data3d.isel(lon=slice(0, 1)), patterns, weights1d)


def test_regime_index(data3d):
    mean = data3d.mean(dim=["lat", "lon"])
    std = data3d.std(dim=["lat", "lon"])
    result = regime_index(data3d, mean, std)
    assert result.dims == data3d.dims
    assert result.shape == data3d.shape
    np.testing.assert_allclose(result, (data3d - mean) / std)
