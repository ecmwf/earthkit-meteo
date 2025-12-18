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


# The type of the input per level is encoded in the test name as three letters with:
#   s: scalar
#   a: array
#
# So, e.g. "s_a_s" means the following on a level:
#  - value is scalar
#  - pres is array
#  - target is scalar


def make_xr(xp, data, coord):
    import xarray as xr

    d = xr.DataArray(
        data,
        dims=("level",),
        coords={"level": coord},
    )

    ds = xr.Dataset({"data": d})
    return ds


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "value,pres,target,mode,expected_value",
    [
        (
            [1012.0, 1000.0, 990.0],
            [1012.0, 1000.0, 990.0],
            [1022.0, 1012.0, 1009.0, 995.0, 990.0, 987.0],
            "linear",
            [np.nan, 1012.0, 1009, 995, 990.0, np.nan],
        ),
        (
            [1012.0, np.nan, 990.0],
            [1012.0, 1000.0, 990.0],
            [1022.0, 1012.0, 1009.0, 995.0, 990.0, 987.0],
            "linear",
            [np.nan, 1012.0, np.nan, np.nan, np.nan, np.nan],
        ),
        (
            [1012.0, np.nan, 990.0],
            [1012.0, 1000.0, 990.0],
            [1022.0, 1012.0, 1009.0, 995.0, 990.0, 987.0],
            "linear",
            [np.nan, 1012.0, np.nan, np.nan, np.nan, np.nan],
        ),
        (
            [
                990.0,
                1000.0,
                1012.0,
            ],
            [
                990.0,
                1000.0,
                1012.0,
            ],
            [1022.0, 1009.0, 995.0, 987.0],
            "linear",
            [np.nan, 1009, 995, np.nan],
        ),
        (
            [1012.0, 1000.0, 990.0],
            [1012.0, 1000.0, 990.0],
            [1022.0, 1009.5, 1009.0, 1002.0, 1000.0, 995.0, 987.0],
            "log",
            [np.nan, 1009.5117769371, 1009.0133929430, 1002.0099668454, 1000.0, 995.0125628669, np.nan],
        ),
        (
            [1012.0, 1000.0, 990.0],
            [1012.0, 1000.0, 990.0],
            [1022.0, 1009.0, 995.0, 992.0, 987.0],
            "nearest",
            [1012.0, 1012, 1000.0, 990.0, 990.0],
        ),
    ],
)
def test_to_pressure_s_s_s(value, pres, target, mode, expected_value, xp, device):
    """Test to_pressure with scalar value, scalar pres, scalar target"""
    value = xp.asarray(value, device=device)
    pres = xp.asarray(pres, device=device)
    target = xp.asarray(target, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.interpolate_monotonic(value, pres, target, mode)
    assert xp.allclose(r, expected_value, equal_nan=True)


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
        (
            [200.0, 100.0],
            [900.0, 1000.0],
            [[1000.0, 900.0, 1000.0], [800.0, 700.0, 600.0]],
            "linear",
        ),
    ],
)
def test_to_pressure_s_s_a(value, pres, target, mode, xp, device):
    """Test to_pressure with scalar value, scalar pres, scalar target"""
    value = xp.asarray(value, device=device)
    pres = xp.asarray(pres, device=device)
    target = xp.asarray(target, device=device)

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
def test_to_pressure_s_a_s(value, pres, target, mode, xp, device):
    """Test to_pressure with scalar value, scalar pres, scalar target"""
    value = xp.asarray(value, device=device)
    pres = xp.asarray(pres, device=device)
    target = xp.asarray(target, device=device)

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
def test_to_pressure_s_a_a(value, pres, target, mode, xp, device):
    """Test to_pressure with scalar value, scalar pres, scalar target"""
    value = xp.asarray(value, device=device)
    pres = xp.asarray(pres, device=device)
    target = xp.asarray(target, device=device)

    with pytest.raises(ValueError):
        vertical.interpolate_monotonic(value, pres, target, mode)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "value,pres,target,mode,expected_value",
    [
        (
            [[1020.0, 1010.0, 1000.0], [920.0, 910.0, 900.0], [820, 810.0, 800.0]],
            [[1020.0, 1010.0, 1000.0], [920.0, 910.0, 900.0], [820, 810.0, 800.0]],
            [1030.0, 1018.0, 1005.0, 950.0, 914.0, 905.0, 850.0, 814.0, 805.0, 790.0],
            "linear",
            [
                [np.nan, np.nan, np.nan],
                [1018.0, np.nan, np.nan],
                [1005.0, 1005.0, np.nan],
                [950.0, 950.0, 950.0],
                [914.0, 914.0, 914.0],
                [905.0, 905.0, 905.0],
                [850.0, 850.0, 850.0],
                [np.nan, 814.0, 814.0],
                [np.nan, np.nan, 805.0],
                [np.nan, np.nan, np.nan],
            ],
        ),
        # (
        #     [[820, 810.0, 800.0], [920.0, 910.0, 900.0], [1020.0, 1010.0, 1000.0]],
        #     [[820, 810.0, 800.0], [920.0, 910.0, 900.0], [1020.0, 1010.0, 1000.0]],
        #     [1030.0, 1018.0, 1005.0, 950.0, 914.0, 905.0, 850.0, 814.0, 805.0, 790.0],
        #     "linear",
        #     [
        #         [np.nan, np.nan, np.nan],
        #         [1018.0, np.nan, np.nan],
        #         [1005.0, 1005.0, np.nan],
        #         [950.0, 950.0, 950.0],
        #         [914.0, 914.0, 914.0],
        #         [905.0, 905.0, 905.0],
        #         [850.0, 850.0, 850.0],
        #         [np.nan, 814.0, 814.0],
        #         [np.nan, np.nan, 805.0],
        #         [np.nan, np.nan, np.nan],
        #     ],
        # ),
    ],
)
def test_to_pressure_a_a_s(value, pres, target, mode, expected_value, xp, device):
    """Test to_pressure with array value, array pres, scalar target"""
    value = xp.asarray(value, device=device)
    pres = xp.asarray(pres, device=device)
    target = xp.asarray(target, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    # x = make_xr(value, pres)
    # print(x)

    r = vertical.interpolate_monotonic(value, pres, target, mode)
    assert xp.allclose(r, expected_value, equal_nan=True)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "value,pres,target,mode,expected_value",
    [
        (
            [[200.0, 210.0, 220.0], [100, 110, 120], [0, 10.0, 20.0]],
            [1000.0, 900.0, 800.0],
            [1020.0, 1000.0, 960.0, 900.0, 860.0, 800.0, 750.0],
            "linear",
            [
                [np.nan, np.nan, np.nan],
                [200.0, 210.0, 220.0],
                [160.0, 170.0, 180.0],
                [100.0, 110.0, 120.0],
                [60.0, 70.0, 80.0],
                [0.0, 10.0, 20.0],
                [np.nan, np.nan, np.nan],
            ],
        ),
        (
            [[0, 10.0, 20.0], [100, 110, 120], [200.0, 210.0, 220.0]],
            [
                800.0,
                900.0,
                1000.0,
            ],
            [1020.0, 1000.0, 960.0, 900.0, 860.0, 800.0, 750.0],
            # [1000.0],
            "linear",
            [
                [np.nan, np.nan, np.nan],
                [200.0, 210.0, 220.0],
                [160.0, 170.0, 180.0],
                [100.0, 110.0, 120.0],
                [60.0, 70.0, 80.0],
                [0.0, 10.0, 20.0],
                [np.nan, np.nan, np.nan],
            ],
        ),
    ],
)
def test_to_pressure_a_s_s(value, pres, target, mode, expected_value, xp, device):
    """Test to_pressure with array value, array pres, scalar target"""
    value = xp.asarray(value, device=device)
    pres = xp.asarray(pres, device=device)
    target = xp.asarray(target, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.interpolate_monotonic(value, pres, target, mode)
    assert xp.allclose(r, expected_value, equal_nan=True)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "value,pres,target,mode,expected_value",
    [
        (
            [[200.0, 210.0, 220.0], [100, 110, 120], [0, 10.0, 20.0]],
            [1000.0, 900.0, 800.0],
            [
                [1030.0, 1020.0, 1010.0],
                [1020.0, 1000.0, 1000.0],
                [960.0, 900.0, 900.0],
                [860.0, 800.0, 800.0],
                [750.0, 800.0, 800.0],
                [749.0, 750.0, 700],
            ],
            "linear",
            [
                [np.nan, np.nan, np.nan],
                [np.nan, 210.0, 220.0],
                [160.0, 110.0, 120.0],
                [60.0, 10.0, 20.0],
                [np.nan, 10.0, 20.0],
                [np.nan, np.nan, np.nan],
            ],
        ),
        (
            [[0, 10.0, 20.0], [100, 110, 120], [200.0, 210.0, 220.0]],
            [800.0, 900.0, 1000.0],
            [
                [1030.0, 1020.0, 1010.0],
                [1020.0, 1000.0, 1000.0],
                [960.0, 900.0, 900.0],
                [860.0, 800.0, 800.0],
                [750.0, 800.0, 800.0],
                [749.0, 750.0, 700],
            ],
            "linear",
            [
                [np.nan, np.nan, np.nan],
                [np.nan, 210.0, 220.0],
                [160.0, 110.0, 120.0],
                [60.0, 10.0, 20.0],
                [np.nan, 10.0, 20.0],
                [np.nan, np.nan, np.nan],
            ],
        ),
    ],
)
def test_to_pressure_a_s_a(value, pres, target, mode, expected_value, xp, device):
    """Test to_pressure with array value, array pres, array target"""
    value = xp.asarray(value, device=device)
    pres = xp.asarray(pres, device=device)
    target = xp.asarray(target, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.interpolate_monotonic(value, pres, target, mode)
    assert xp.allclose(r, expected_value, equal_nan=True)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "value,pres,target,mode,expected_value",
    [
        (
            [[200.0, 210.0, 220.0], [100, 110, 120], [0, 10.0, 20.0]],
            [[1020.0, 1010.0, 1000.0], [920.0, 910.0, 900.0], [820, 810.0, 800.0]],
            [
                [1030.0, 1020.0, 1010.0],
                [1020.0, 1000.0, 1000.0],
                [960.0, 900.0, 900.0],
                [860.0, 800.0, 800.0],
                [750.0, 810.0, 800.0],
                [749.0, 750.0, 700],
            ],
            "linear",
            [
                [np.nan, np.nan, np.nan],
                [200.0, 200.0, 220.0],
                [140.0, 100.0, 120.0],
                [40.0, np.nan, 20.0],
                [np.nan, 10.0, 20.0],
                [np.nan, np.nan, np.nan],
            ],
        ),
        (
            [[0, 10.0, 20.0], [100, 110, 120], [200.0, 210.0, 220.0]],
            [[820, 810.0, 800.0], [920.0, 910.0, 900.0], [1020.0, 1010.0, 1000.0]],
            [
                [1030.0, 1020.0, 1010.0],
                [1020.0, 1000.0, 1000.0],
                [960.0, 900.0, 900.0],
                [860.0, 800.0, 800.0],
                [750.0, 810.0, 800.0],
                [749.0, 750.0, 700],
            ],
            "linear",
            [
                [np.nan, np.nan, np.nan],
                [200.0, 200.0, 220.0],
                [140.0, 100.0, 120.0],
                [40.0, np.nan, 20.0],
                [np.nan, 10.0, 20.0],
                [np.nan, np.nan, np.nan],
            ],
        ),
    ],
)
def test_to_pressure_a_a_a(value, pres, target, mode, expected_value, xp, device):
    """Test to_pressure with array value, array pres, array target"""
    value = xp.asarray(value, device=device)
    pres = xp.asarray(pres, device=device)
    target = xp.asarray(target, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.interpolate_monotonic(value, pres, target, mode)
    assert xp.allclose(r, expected_value, equal_nan=True)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "value,pres,target,mode,expected_value",
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
def test_to_pressure_s_s_s_aux(value, pres, target, mode, expected_value, xp, device):
    """Test to_pressure with scalar value, scalar pres, scalar target"""
    value = xp.asarray(value, device=device)
    pres = xp.asarray(pres, device=device)
    target = xp.asarray(target, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.interpolate_monotonic(
        data=value,
        coord=pres,
        target_coord=target,
        interpolation=mode,
        aux_max_level_data=1100,
        aux_max_level_coord=1100,
    )
    assert xp.allclose(r, expected_value, equal_nan=True)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "value,h,target,mode,expected_value",
    [
        (
            [-100.0, 1.0, 100],
            [-100.0, 1.0, 100],
            [-200.0, -100.0, 0.0, 1.0, 50.0, 100.0, 150.0],
            "linear",
            [np.nan, -100.0, 0.0, 1.0, 50.0, 100.0, np.nan],
        ),
    ],
)
def test_to_height_s_s_s(value, h, target, mode, expected_value, xp, device):
    """Test to_pressure with scalar value, scalar pres, scalar target"""
    value = xp.asarray(value, device=device)
    h = xp.asarray(h, device=device)
    target = xp.asarray(target, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.interpolate_monotonic(data=value, coord=h, target_coord=target, interpolation=mode)
    assert xp.allclose(r, expected_value, equal_nan=True)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "value,h,target,mode,expected_value",
    [
        (
            [[200.0, 210.0, 220.0], [100, 110, 120], [0, 10.0, 20.0]],
            [[10.0, 20.0, 100.0], [110.0, 120.0, 200.0], [210.0, 220.0, 300.0]],
            [
                [-100.0, 1.0, 10.0],
                [-100.0, 30.0, 10.0],
                [10.0, 20.0, 100.0],
                [20.0, 30.0, 110.0],
                [120.0, 130.0, 210.0],
                [50.0, 130.0, 150.0],
                [220.0, 130.0, 320.0],
                [220.0, 230.0, 320.0],
            ],
            "linear",
            [
                [np.nan, np.nan, np.nan],
                [np.nan, 200.0, np.nan],
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
def test_to_height_a_a_a(value, h, target, mode, expected_value, xp, device):
    """Test to_pressure with array value, array pres, array target"""
    value = xp.asarray(value, device=device)
    h = xp.asarray(h, device=device)
    target = xp.asarray(target, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.interpolate_monotonic(
        data=value,
        coord=h,
        target_coord=target,
        interpolation=mode,
    )
    assert xp.allclose(r, expected_value, equal_nan=True)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "value,h,target,mode,expected_value",
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
def test_to_height_s_s_s_aux(value, h, target, mode, expected_value, xp, device):
    """Test to_pressure with scalar value, scalar pres, scalar target"""
    value = xp.asarray(value, device=device)
    h = xp.asarray(h, device=device)
    target = xp.asarray(target, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.interpolate_monotonic(
        data=value,
        coord=h,
        target_coord=target,
        interpolation=mode,
        aux_min_level_data=0.0,
        aux_min_level_coord=0.0,
    )
    assert xp.allclose(r, expected_value, equal_nan=True)


@pytest.mark.parametrize("xp, device", NUMPY)
@pytest.mark.parametrize(
    "value,h,target,aux_value,aux_coord,mode,expected_value",
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
def test_to_height_a_a_a_aux(value, h, target, aux_value, aux_coord, mode, expected_value, xp, device):
    """Test to_pressure with array value, array pres, array target"""
    value = xp.asarray(value, device=device)
    h = xp.asarray(h, device=device)
    target = xp.asarray(target, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.interpolate_monotonic(
        data=value,
        coord=h,
        target_coord=target,
        interpolation=mode,
        aux_min_level_data=aux_value,
        aux_min_level_coord=aux_coord,
    )
    assert xp.allclose(r, expected_value, equal_nan=True)
