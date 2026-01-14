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
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo import vertical

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})

NUMPY = [x for x in NAMESPACE_DEVICES if x[0]._earthkit_array_namespace_name == "numpy"]


# The type of the input data per level is encoded in the test name as three letters with:
#   s: scalar
#   a: array
#
# So, e.g. "s_a_s" means the following input data on a level:
#  - data is scalar
#  - coord is array
#  - target_coord is scalar


def _get_data():
    import os
    import sys

    here = os.path.dirname(__file__)
    sys.path.insert(0, here)

    from _monotonic_cases import cases

    return cases


DATA = _get_data()


def _check_array_interpolate_monotonic(data, coord, target_coord, mode, expected_data, xp, device):
    """Test to_pressure with scalar value, scalar pres, scalar target"""
    data = xp.asarray(data, device=device)
    coord = xp.asarray(coord, device=device)
    target_coord = xp.asarray(target_coord, device=device)
    expected_data = xp.asarray(expected_data, device=device)

    r = vertical.interpolate_monotonic(data, coord, target_coord, mode)
    assert xp.allclose(r, expected_data, equal_nan=True)


# @pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "data,coord,target_coord,mode,expected_data",
    DATA["pressure_s_s_s"],
)
def test_array_interpolate_monotonic_s_s_s_1(data, coord, target_coord, mode, expected_data, xp, device):
    """Test with scalar data, scalar coord, scalar target_coord"""
    _check_array_interpolate_monotonic(data, coord, target_coord, mode, expected_data, xp, device)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "data,coord,target_coord,mode,expected_data",
    DATA["height_s_s_s"],
)
def test_array_interpolate_monotonic_s_s_s_2(data, coord, target_coord, mode, expected_data, xp, device):
    """Test with scalar data, scalar coord, scalar target_coord"""
    _check_array_interpolate_monotonic(data, coord, target_coord, mode, expected_data, xp, device)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "data,coord,target_coord,mode,expected_data",
    DATA["pressure_a_a_s"],
)
def test_array_interpolate_monotonic_a_a_s(data, coord, target_coord, mode, expected_data, xp, device):
    """Test with array data, array coord, scalar target_coord"""
    _check_array_interpolate_monotonic(data, coord, target_coord, mode, expected_data, xp, device)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "data,coord,target_coord,mode,expected_data",
    DATA["pressure_a_s_s"],
)
def test_array_interpolate_monotonic_a_s_s(data, coord, target_coord, mode, expected_data, xp, device):
    """Test with array data, scalar coord, scalar target_coord"""
    _check_array_interpolate_monotonic(data, coord, target_coord, mode, expected_data, xp, device)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "data,coord,target_coord,mode,expected_data",
    DATA["pressure_a_s_a"],
)
def test_array_interpolate_monotonic_a_s_a(data, coord, target_coord, mode, expected_data, xp, device):
    """Test with array data, scalar coord, array target_coord"""
    _check_array_interpolate_monotonic(data, coord, target_coord, mode, expected_data, xp, device)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "data,coord,target_coord,mode,expected_data",
    DATA["pressure_a_a_a"],
)
def test_array_interpolate_monotonic_a_a_a_1(data, coord, target_coord, mode, expected_data, xp, device):
    """Test with array data, array coord, array target_coord"""
    _check_array_interpolate_monotonic(data, coord, target_coord, mode, expected_data, xp, device)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "data,coord,target_coord,mode,expected_data",
    DATA["height_a_a_a"],
)
def test_array_interpolate_monotonic_a_a_a_2(data, coord, target_coord, mode, expected_data, xp, device):
    """Test with array data, array coord, array target_coord"""
    _check_array_interpolate_monotonic(data, coord, target_coord, mode, expected_data, xp, device)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "value,pres,target,mode",
    [
        (
            [100.0, 200.0],
            [1000.0, 900.0],
            [[1000.0, 900.0, 1000.0], [800.0, 700.0, 600.0]],
            "linear",
        ),
    ],
)
def test_array_interpolate_monotonic_s_s_a(value, pres, target, mode, xp, device):
    value = xp.asarray(value, device=device)
    pres = xp.asarray(pres, device=device)
    target = xp.asarray(target, device=device)

    # s_s_a is not supported
    with pytest.raises(ValueError):
        vertical.interpolate_monotonic(value, pres, target, mode)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "value,pres,target,mode",
    [
        (
            [100.0, 200.0],
            [[1000.0, 900.0, 1000.0], [800.0, 700.0, 600.0]],
            [1000.0, 900.0],
            "linear",
        ),
    ],
)
def test_array_interpolate_monotonic_s_a_s(value, pres, target, mode, xp, device):
    value = xp.asarray(value, device=device)
    pres = xp.asarray(pres, device=device)
    target = xp.asarray(target, device=device)

    # s_a_s is not supported
    with pytest.raises(ValueError):
        vertical.interpolate_monotonic(value, pres, target, mode)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "value,pres,target,mode",
    [
        (
            [100.0, 200.0],
            [[1000.0, 900.0, 1000.0], [800.0, 700.0, 600.0]],
            [[1000.0, 900.0, 1000.0], [800.0, 700.0, 600.0], [700.0, 600.0, 500.0]],
            "linear",
        ),
    ],
)
def test_array_interpolate_monotonic_s_a_a(value, pres, target, mode, xp, device):
    value = xp.asarray(value, device=device)
    pres = xp.asarray(pres, device=device)
    target = xp.asarray(target, device=device)

    # s_a_a is not supported
    with pytest.raises(ValueError):
        vertical.interpolate_monotonic(value, pres, target, mode)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "data,coord,target_coord,mode,expected_data",
    [
        (
            [1012.0, 1000.0, 990.0],
            [1012.0, 1000.0, 990.0],
            [1101.0, 1100.0, 1022.0, 1012.0, 1009.0, 995.0, 990.0, 987.0],
            "linear",
            [np.nan, 1100.0, 1022.0, 1012.0, 1009.0, 995.0, 990.0, np.nan],
        ),
    ],
)
def test_array_interpolate_monotonic_to_pressure_s_s_s_aux(
    data, coord, target_coord, mode, expected_data, xp, device
):
    """Test interpolation with auxiliary min/max level data"""
    data = xp.asarray(data, device=device)
    coord = xp.asarray(coord, device=device)
    target_coord = xp.asarray(target_coord, device=device)
    expected_data = xp.asarray(expected_data, device=device)

    # prescribe aux level at the bottom (max pressure in input is 1012 hPa)
    r = vertical.interpolate_monotonic(
        data=data,
        coord=coord,
        target_coord=target_coord,
        interpolation=mode,
        aux_max_level_data=1100,
        aux_max_level_coord=1100,
    )
    assert xp.allclose(r, expected_data, equal_nan=True)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "data,coord,target_coord,mode,expected_data",
    [
        (
            [10.0, 20.0, 100],
            [10.0, 20.0, 100],
            [-1.0, 0.0, 4.0, 10.0, 50.0, 100.0, 150.0],
            "linear",
            [np.nan, 0.0, 4.0, 10.0, 50.0, 100.0, np.nan],
        ),
    ],
)
def test_array_interpolate_monotonic_to_height_s_s_s_aux(
    data, coord, target_coord, mode, expected_data, xp, device
):
    """Test interpolation with auxiliary min/max level data"""
    data = xp.asarray(data, device=device)
    coord = xp.asarray(coord, device=device)
    target_coord = xp.asarray(target_coord, device=device)
    expected_data = xp.asarray(expected_data, device=device)

    # prescribe aux level at the bottom (min height in input is 10.0 m)
    r = vertical.interpolate_monotonic(
        data=data,
        coord=coord,
        target_coord=target_coord,
        interpolation=mode,
        aux_min_level_data=0.0,
        aux_min_level_coord=0.0,
    )
    assert xp.allclose(r, expected_data, equal_nan=True)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "data,coord,target_coord,aux_data,aux_coord,mode,expected_data",
    [
        (
            [[200.0, 210.0, 220.0], [100, 110, 120], [0, 10.0, 20.0]],
            [[10.0, 20.0, 100.0], [110.0, 120.0, 200.0], [210.0, 220.0, 300.0]],
            [
                [-100.0, 2.0, 10.0],
                [-100.0, 30.0, 10.0],
                [10.0, 20.0, 100.0],
                [20.0, 30.0, 110.0],
                [120.0, 130.0, 210.0],
                [50.0, 130.0, 150.0],
                [220.0, 130.0, 320.0],
                [220.0, 230.0, 320.0],
            ],
            [300.0, 310.0, 320.0],
            [0.0, 0.0, 0.0],
            "linear",
            [
                [np.nan, 300.0, 310.0],
                [np.nan, 200.0, 310.0],
                [200.0, 210.0, 220.0],
                [190.0, 200.0, 210.0],
                [90.0, 100.0, 110.0],
                [160.0, 100.0, 170.0],
                [np.nan, 100.0, np.nan],
                [np.nan, np.nan, np.nan],
            ],
        ),
    ],
)
def test_array_interpolate_monotonic_to_height_a_a_a_aux(
    data, coord, target_coord, aux_data, aux_coord, mode, expected_data, xp, device
):
    """Test interpolation with auxiliary min/max level data"""
    data = xp.asarray(data, device=device)
    coord = xp.asarray(coord, device=device)
    target_coord = xp.asarray(target_coord, device=device)
    expected_data = xp.asarray(expected_data, device=device)

    # prescribe aux level at the bottom
    r = vertical.interpolate_monotonic(
        data=data,
        coord=coord,
        target_coord=target_coord,
        interpolation=mode,
        aux_min_level_data=aux_data,
        aux_min_level_coord=aux_coord,
    )
    assert xp.allclose(r, expected_data, equal_nan=True)
