# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest

from earthkit.meteo import vertical


@pytest.mark.parametrize(
    "z,expected_value",
    [(0, 0), (1000, 101.97162129779284), ([1000, 10000], [101.9716212978, 1019.7162129779])],
)
def test_geopotential_height_from_geopotential(z, expected_value):
    if isinstance(z, list):
        z = np.asarray(z)

    r = vertical.geopotential_height_from_geopotential(z)

    r = np.asarray(r)
    expected_value = np.asarray(expected_value)
    assert np.allclose(r, expected_value)


@pytest.mark.parametrize(
    "h,expected_value",
    [
        (0, 0),
        (5102.664476187331, 50000),
        ([1019.8794448450, 5102.6644761873, 7146.0195417809], [10000, 50000, 70000]),
    ],
)
def test_geopotential_from_geometric_height(h, expected_value):
    if isinstance(h, list):
        h = np.asarray(h)

    r = vertical.geopotential_from_geometric_height(h)

    r = np.asarray(r)
    expected_value = np.asarray(expected_value)
    assert np.allclose(r, expected_value)


@pytest.mark.parametrize(
    "h,expected_value",
    [
        (0, 0),
        (5003.9269715243, 5000),
        ([1000.1569802279, 5003.9269715243, 7007.6992829768], [1000, 5000, 7000]),
    ],
)
def test_geopotential_height_from_geometric_height(h, expected_value):
    if isinstance(h, list):
        h = np.asarray(h)

    r = vertical.geopotential_height_from_geometric_height(h)

    r = np.asarray(r)
    expected_value = np.asarray(expected_value)
    assert np.allclose(r, expected_value)


@pytest.mark.parametrize(
    "z,expected_value",
    [
        (0, 0),
        (50000, 5102.664476187331),
        ([10000, 50000, 70000], [1019.8794448450, 5102.6644761873, 7146.0195417809]),
    ],
)
def test_geometric_height_from_geopotential(z, expected_value):
    if isinstance(z, list):
        z = np.asarray(z)

    r = vertical.geometric_height_from_geopotential(z)

    r = np.asarray(r)
    expected_value = np.asarray(expected_value)
    assert np.allclose(r, expected_value)


@pytest.mark.parametrize(
    "zh,expected_value",
    [
        (0, 0),
        (5000, 5003.9269715243),
        ([1000, 5000, 7000], [1000.1569802279, 5003.9269715243, 7007.6992829768]),
    ],
)
def test_geometric_height_from_geopotential_height(zh, expected_value):
    if isinstance(zh, list):
        zh = np.asarray(zh)

    r = vertical.geometric_height_from_geopotential_height(zh)

    r = np.asarray(r)
    expected_value = np.asarray(expected_value)
    assert np.allclose(r, expected_value)
