# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from importlib import import_module

import numpy as np
import pytest
from earthkit.utils.array.namespace import _NUMPY_NAMESPACE
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo import vertical
from earthkit.meteo.utils.testing import Tolerance

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


def _get_data_1(name):
    import os
    import sys

    here = os.path.dirname(__file__)
    sys.path.insert(0, here)

    return import_module(name)


DATA_HYBRID_CORE = _get_data_1("_hybrid_core_data")
DATA_HYBRID_H = _get_data_1("_hybrid_height_data")
DATA_PL = _get_data_1("_pl_data")


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


@pytest.mark.parametrize("xp, device", [(_NUMPY_NAMESPACE, "cpu")])
@pytest.mark.parametrize(
    "_kwargs,expected_values",
    [
        (
            {"target_p": [85000.0, 50000.0], "interpolation": "linear"},
            [[263.50982741, 287.70299692], [238.00383748, 259.50822691]],
        ),
        (
            {"target_p": [85000.0, 50000.0, 95100.0], "interpolation": "linear"},
            [[263.50982741, 287.70299692], [238.00383748, 259.50822691], [np.nan, 292.08454145]],
        ),
        (
            {
                "target_p": [85000.0, 50000.0, 95100.0],
                "aux_bottom_p": [95178.337944, 102659.81019512],
                "aux_bottom_data": [270.0, 293.0],
                "interpolation": "linear",
            },
            [[263.50982741, 287.70299692], [238.00383748, 259.50822691], [269.21926951, 292.08454145]],
        ),
        (
            {
                "target_p": [95100.0],
                "interpolation": "linear",
            },
            [np.nan, 292.08454145],
        ),
        (
            {
                "target_p": [95100.0],
                "aux_bottom_p": [95178.337944, 102659.81019512],
                "aux_bottom_data": [270.0, 293.0],
                "interpolation": "linear",
            },
            [269.21926951, 292.08454145],
        ),
    ],
)
@pytest.mark.parametrize("part", [None, slice(-50, None)])
def test_array_interpolate_hybrid_to_pressure_levels(_kwargs, expected_values, part, xp, device):
    r_ref = expected_values

    A, B = vertical.hybrid_level_parameters(137, model="ifs")
    A = A.tolist()
    B = B.tolist()

    sp = DATA_HYBRID_H.p_surf
    t = DATA_HYBRID_H.t

    t, r_ref, sp, A, B = (xp.asarray(x, device=device) for x in [t, r_ref, sp, A, B])

    if part:
        t = t[part]

    _kwargs = dict(_kwargs)
    target_p = _kwargs.pop("target_p")

    r = vertical.interpolate_hybrid_to_pressure_levels(
        t,  # data to interpolate
        target_p,
        A,
        B,
        sp,
        **_kwargs,
    )

    # print(repr(r))

    tolerance = Tolerance({64: (1e-8, 1e-6), 32: (10, 1e-6)})
    atol, rtol = tolerance.get(dtype=t.dtype)
    assert xp.allclose(
        r, r_ref, atol=atol, rtol=rtol, equal_nan=True
    ), f"max abs diff={xp.max(xp.abs(r - r_ref))}"


@pytest.mark.parametrize("xp, device", [(_NUMPY_NAMESPACE, "cpu")])
@pytest.mark.parametrize(
    "_kwargs,expected_values",
    [
        (
            {
                "target_h": [1000.0, 5000.0],
                "h_type": "geometric",
                "h_reference": "ground",
                "interpolation": "linear",
            },
            [[262.3693894784, 291.4452034379], [236.7746100208, 265.4952859218]],
        ),
        (
            {
                "target_h": [1000.0, 5000.0],
                "h_type": "geometric",
                "h_reference": "sea",
                "interpolation": "linear",
            },
            [[265.8344752939, 291.0419484632], [239.8099274052, 264.9629089069]],
        ),
        (
            {
                "target_h": [1000.0, 5000.0],
                "h_type": "geopotential",
                "h_reference": "ground",
                "interpolation": "linear",
            },
            [[262.3657943604, 291.4447171210], [236.7517288039, 265.4713984425]],
        ),
        (
            {
                "target_h": [1000.0, 5000.0],
                "h_type": "geopotential",
                "h_reference": "sea",
                "interpolation": "linear",
            },
            [[265.8333668681, 291.0411459042], [239.7860545644, 264.9382000331]],
        ),
        (
            {
                "target_h": [1000.0, 5000.0, 5.0],
                "h_type": "geometric",
                "h_reference": "ground",
                "interpolation": "linear",
            },
            [[262.3693894784, 291.4452034379], [236.7746100208, 265.4952859218], [np.nan, np.nan]],
        ),
        (
            {
                "target_h": [1000.0, 5000.0, 5.0],
                "h_type": "geometric",
                "h_reference": "ground",
                "aux_bottom_h": 0.0,
                "aux_bottom_data": [280.0, 300.0],
                "interpolation": "linear",
            },
            [
                [262.3693894784, 291.4452034379],
                [236.7746100208, 265.4952859218],
                [274.0481585682, 296.5000734836],
            ],
        ),
        (
            {
                "target_h": [5.0],
                "h_type": "geometric",
                "h_reference": "ground",
                "interpolation": "linear",
            },
            [np.nan, np.nan],
        ),
        (
            {
                "target_h": [5.0],
                "h_type": "geometric",
                "h_reference": "ground",
                "aux_bottom_h": 0.0,
                "aux_bottom_data": [280.0, 300.0],
                "interpolation": "linear",
            },
            [274.0481585682, 296.5000734836],
        ),
    ],
)
@pytest.mark.parametrize("part", [None, slice(-50, None)])
def test_array_interpolate_hybrid_to_height_levels(_kwargs, expected_values, part, xp, device):
    r_ref = expected_values

    A, B = vertical.hybrid_level_parameters(137, model="ifs")
    A = A.tolist()
    B = B.tolist()

    sp = DATA_HYBRID_H.p_surf
    t = DATA_HYBRID_H.t
    q = DATA_HYBRID_H.q
    zs = DATA_HYBRID_H.z_surf

    r_ref, t, q, zs, sp, A, B = (xp.asarray(x, device=device) for x in [r_ref, t, q, zs, sp, A, B])

    if part:
        t = t[part]
        q = q[part]

    _kwargs = dict(_kwargs)
    target_h = _kwargs.pop("target_h")

    r = vertical.interpolate_hybrid_to_height_levels(
        t,  # data to interpolate
        target_h,
        t,
        q,
        zs,
        A,
        B,
        sp,
        **_kwargs,
    )

    # print(repr(r))

    tolerance = Tolerance({64: (1e-8, 1e-6), 32: (10, 1e-6)})
    atol, rtol = tolerance.get(dtype=t.dtype)
    assert xp.allclose(
        r, r_ref, atol=atol, rtol=rtol, equal_nan=True
    ), f"max abs diff={xp.max(xp.abs(r - r_ref))}"


@pytest.mark.parametrize("xp, device", [(_NUMPY_NAMESPACE, "cpu")])
@pytest.mark.parametrize(
    "_kwargs,expected_values",
    [
        (
            {
                "target_h": [1000.0, 5000.0],
                "h_type": "geometric",
                "h_reference": "ground",
                "interpolation": "linear",
            },
            [[294.20429573, 299.22387254], [271.02124509, 272.90306903]],
        ),
        (
            {
                "target_h": [1000.0, 5000.0],
                "h_type": "geometric",
                "h_reference": "sea",
                "interpolation": "linear",
            },
            [[298.4516800756, 298.9948524649], [274.0326115423, 272.7133842002]],
        ),
        (
            {
                "target_h": [1000.0, 5000.0],
                "h_type": "geopotential",
                "h_reference": "ground",
                "interpolation": "linear",
            },
            [[294.2022875759, 299.2221015691], [270.9937963229, 272.8758976631]],
        ),
        (
            {
                "target_h": [1000.0, 5000.0],
                "h_type": "geopotential",
                "h_reference": "sea",
                "interpolation": "linear",
            },
            [[298.4498485083, 298.9930209085], [274.0092073632, 272.6859411558]],
        ),
        (
            {
                "target_h": [1000.0, 5000.0, 2.0],
                "h_type": "geometric",
                "h_reference": "ground",
                "interpolation": "linear",
            },
            [
                [
                    294.20429573,
                    299.22387254,
                ],
                [271.02124509, 272.90306903],
                [302.0918370407, np.nan],
            ],
        ),
        (
            {
                "target_h": [1000.0, 5000.0, 2.0],
                "h_type": "geometric",
                "h_reference": "ground",
                "interpolation": "linear",
                "aux_bottom_h": 0.0,
                "aux_bottom_data": [304.0, 306.0],
            },
            [
                [
                    294.20429573,
                    299.22387254,
                ],
                [271.02124509, 272.90306903],
                [302.0918370407, 305.9996224989],
            ],
        ),
        (
            {
                "target_h": [2.0],
                "h_type": "geometric",
                "h_reference": "ground",
                "interpolation": "linear",
            },
            [302.0918370407, np.nan],
        ),
        (
            {
                "target_h": [2.0],
                "h_type": "geometric",
                "h_reference": "ground",
                "interpolation": "linear",
                "aux_bottom_h": 0.0,
                "aux_bottom_data": [304.0, 306.0],
            },
            [302.0918370407, 305.9996224989],
        ),
    ],
)
def test_array_interpolate_pressure_to_height_levels(_kwargs, expected_values, xp, device):
    r_ref = expected_values

    t = DATA_PL.t  # temperature [K]
    z = DATA_PL.z  # geopotential [m2/s2]
    sp = DATA_PL.p_surf  # surface pressure [Pa]
    zs = DATA_PL.z_surf  # surface geopotential [m2/s2]

    t, z, r_ref, sp, zs = (xp.asarray(x, device=device) for x in [t, z, r_ref, sp, zs])

    _kwargs = dict(_kwargs)
    target_h = _kwargs.pop("target_h")

    r = vertical.interpolate_pressure_to_height_levels(
        t,  # data to interpolate
        target_h,
        z,
        zs,
        **_kwargs,
    )

    tolerance = Tolerance({64: (1e-8, 1e-6), 32: (10, 1e-6)})
    atol, rtol = tolerance.get(dtype=t.dtype)
    assert xp.allclose(
        r, r_ref, atol=atol, rtol=rtol, equal_nan=True
    ), f"max abs diff={xp.max(xp.abs(r - r_ref))}"
