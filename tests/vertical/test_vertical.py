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
from earthkit.utils.array.namespace import _NUMPY_NAMESPACE
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo import vertical

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


def _get_data():
    import os
    import sys

    here = os.path.dirname(__file__)
    sys.path.insert(0, here)
    import _vertical_data

    return _vertical_data


DATA = _get_data()


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "z,expected_value",
    [(0.0, 0.0), (1000.0, 101.97162129779284), ([1000.0, 10000.0], [101.9716212978, 1019.7162129779])],
)
def test_geopotential_height_from_geopotential(z, expected_value, xp, device):
    z = xp.asarray(z, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.geopotential_height_from_geopotential(z)
    assert xp.allclose(r, expected_value)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "h,expected_value",
    [(0.0, 0.0), (101.97162129779284, 1000.0), ([101.9716212978, 1019.7162129779], [1000.0, 10000.0])],
)
def test_geopotential_from_geopotential_height(h, expected_value, xp, device):
    h = xp.asarray(h, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.geopotential_from_geopotential_height(h)
    assert xp.allclose(r, expected_value)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "h,expected_value",
    [
        (0.0, 0.0),
        (5102.664476187331, 50000.0),
        ([1019.8794448450, 5102.6644761873, 7146.0195417809], [10000.0, 50000.0, 70000.0]),
    ],
)
def test_geopotential_from_geometric_height(h, expected_value, xp, device):
    h = xp.asarray(h, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.geopotential_from_geometric_height(h)
    assert xp.allclose(r, expected_value)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "h,expected_value",
    [
        (0.0, 0.0),
        (5003.9269715243, 5000.0),
        ([1000.1569802279, 5003.9269715243, 7007.6992829768], [1000.0, 5000.0, 7000.0]),
    ],
)
def test_geopotential_height_from_geometric_height(h, expected_value, xp, device):
    h = xp.asarray(h, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.geopotential_height_from_geometric_height(h)

    assert xp.allclose(r, expected_value)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "z,expected_value",
    [
        (0.0, 0.0),
        (50000.0, 5102.664476187331),
        ([10000.0, 50000.0, 70000.0], [1019.8794448450, 5102.6644761873, 7146.0195417809]),
    ],
)
def test_geometric_height_from_geopotential(z, expected_value, xp, device):
    z = xp.asarray(z, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.geometric_height_from_geopotential(z)
    assert xp.allclose(r, expected_value)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "zh,expected_value",
    [
        (0.0, 0.0),
        (5000.0, 5003.9269715243),
        ([1000.0, 5000.0, 7000.0], [1000.1569802279, 5003.9269715243, 7007.6992829768]),
    ],
)
def test_geometric_height_from_geopotential_height(zh, expected_value, xp, device):
    zh = xp.asarray(zh, device=device)
    expected_value = xp.asarray(expected_value, device=device)

    r = vertical.geometric_height_from_geopotential_height(zh)
    assert xp.allclose(r, expected_value)


@pytest.mark.parametrize("xp", [_NUMPY_NAMESPACE])
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
def test_pressure_at_model_levels(index, xp):

    sp = DATA.p_surf
    A = DATA.A
    B = DATA.B
    ref_p_full = DATA.p_full
    ref_p_half = DATA.p_half
    ref_delta = DATA.delta
    ref_alpha = DATA.alpha

    sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B = (
        xp.asarray(x) for x in [sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B]
    )

    sp = sp[index[1]]
    ref_p_full = ref_p_full[index]
    ref_p_half = ref_p_half[index]
    ref_delta = ref_delta[index]
    ref_alpha = ref_alpha[index]

    p_full, p_half, delta, alpha = vertical.pressure_at_model_levels(A, B, sp, alpha_top="ifs")

    # print("p_full", repr(p_full))
    # print("p_half", repr(p_half))
    # print("delta", repr(delta))
    # print("alpha", repr(alpha))

    assert xp.allclose(p_full, ref_p_full)
    assert xp.allclose(p_half, ref_p_half)
    assert xp.allclose(delta, ref_delta)
    assert xp.allclose(alpha, ref_alpha)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
def test_relative_geopotential_thickness(index, xp, device):

    A = DATA.A
    B = DATA.B
    alpha = DATA.alpha
    delta = DATA.delta
    t = DATA.t
    q = DATA.q
    z_ref = DATA.z

    z_ref, t, q, alpha, delta, A, B = (
        xp.asarray(x, device=device) for x in [z_ref, t, q, alpha, delta, A, B]
    )

    alpha = alpha[index]
    delta = delta[index]
    t = t[index]
    q = q[index]
    z_ref = z_ref[index]

    z = vertical.relative_geopotential_thickness(alpha, delta, t, q)
    # print("z", repr(z))
    # print("z_ref", repr(z_ref))

    assert xp.allclose(z, z_ref)


@pytest.mark.parametrize("xp", [_NUMPY_NAMESPACE])
@pytest.mark.parametrize(
    "h,p_ref",
    [
        (0, 101183.94696484),
        (1, 101171.8606369517),
        (100.0, 99979.6875841272),
        (5000.0, 53738.035306025726),
        (50000.0, 84.2265165561),
    ],
)
def test_pressure_at_height_levels_all(h, p_ref, xp):
    sp = DATA.p_surf
    A = DATA.A
    B = DATA.B
    t = DATA.t
    q = DATA.q

    sp, h, t, q, A, B = (xp.asarray(x) for x in [sp, h, t, q, A, B])

    sp = sp[0]  # use only the first surface pressure value
    t = t[:, 0]
    q = q[:, 0]

    p = vertical.pressure_at_height_levels(h, t, q, sp, A, B, alpha_top="ifs")
    assert xp.isclose(p, p_ref)


@pytest.mark.parametrize("xp", [_NUMPY_NAMESPACE])
@pytest.mark.parametrize(
    "h,p_ref",
    [
        (0, 101183.94696484),
        (1, 101171.8606369517),
        (100.0, 99979.6875841272),
        (5000.0, 53738.035306025726),
    ],
)
def test_pressure_at_height_levels_part(h, p_ref, xp):
    # get only levels from 90 to 136/137
    part = slice(90, None)

    sp = DATA.p_surf
    A = DATA.A
    B = DATA.B
    t = DATA.t
    q = DATA.q

    assert len(A) == len(B) == len(t) + 1 == len(q) + 1

    sp, h, t, q, A, B = (xp.asarray(x) for x in [sp, h, t, q, A, B])

    sp = sp[0]
    A = A[part]
    B = B[part]
    t = t[part, 0]
    q = q[part, 0]

    p = vertical.pressure_at_height_levels(h, t, q, sp, A, B, alpha_top="ifs")
    assert xp.isclose(p, p_ref)


@pytest.mark.parametrize("xp", [_NUMPY_NAMESPACE])
def test_pressure_at_height_levels_multi_point(xp):
    h = 5000.0  # height in m
    sp = DATA.p_surf
    A = DATA.A
    B = DATA.B
    t = DATA.t
    q = DATA.q

    assert len(A) == len(B) == len(t) + 1 == len(q) + 1
    sp, h, t, q, A, B = (xp.asarray(x) for x in [sp, h, t, q, A, B])

    p_ref = np.array([53738.035306025726, 27290.9128315574])

    p = vertical.pressure_at_height_levels(h, t, q, sp, A, B, alpha_top="ifs")
    assert xp.allclose(p, p_ref)


@pytest.mark.parametrize("xp", [_NUMPY_NAMESPACE])
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
def test_pressure_on_hybrid_levels_1(index, xp):

    sp = DATA.p_surf
    A = DATA.A
    B = DATA.B
    ref_p_full = DATA.p_full
    ref_p_half = DATA.p_half
    ref_delta = DATA.delta
    ref_alpha = DATA.alpha

    sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B = (
        xp.asarray(x) for x in [sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B]
    )

    sp = sp[index[1]]
    ref_p_full = ref_p_full[index]
    ref_p_half = ref_p_half[index]
    ref_delta = ref_delta[index]
    ref_alpha = ref_alpha[index]

    p_full, p_half, delta, alpha = vertical.pressure_on_hybrid_levels(
        A, B, sp, alpha_top="ifs", output=["full", "half", "delta", "alpha"]
    )

    # print("p_full", repr(p_full))
    # print("p_half", repr(p_half))
    # print("delta", repr(delta))
    # print("alpha", repr(alpha))

    assert xp.allclose(p_full, ref_p_full)
    assert xp.allclose(p_half, ref_p_half)
    assert xp.allclose(delta, ref_delta)
    assert xp.allclose(alpha, ref_alpha)


@pytest.mark.parametrize("xp", [_NUMPY_NAMESPACE])
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
@pytest.mark.parametrize(
    "levels", [None, list(range(90, 138)), list(range(137, 90, -1)), [1, 2], [2, 1], [1]]
)
@pytest.mark.parametrize(
    "output",
    [
        "full",
        "half",
        "delta",
        "alpha",
        ["full", "half", "delta", "alpha"],
        ["full", "half"],
        ["half", "full"],
        ["delta", "alpha"],
    ],
)
def test_pressure_on_hybrid_levels_2(index, levels, output, xp):

    sp = DATA.p_surf
    A = DATA.A
    B = DATA.B
    ref_p_full = DATA.p_full
    ref_p_half = DATA.p_half
    ref_delta = DATA.delta
    ref_alpha = DATA.alpha

    # ref_def = {"full": DATA.p_full, "half": DATA.p_half, "delta": DATA.delta, "alpha": DATA.alpha}
    # ref = {
    #     key: val
    #     for key, val in ref_def.items()
    #     if (output == key or (isinstance(output, (list, tuple)) and key in output))
    # }

    # ref_p_full = DATA.p_full
    # ref_p_half = DATA.p_half
    # ref_delta = DATA.delta
    # ref_alpha = DATA.alpha

    sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B = (
        xp.asarray(x) for x in [sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B]
    )

    # sp tests data is 1D
    sp = sp[index[1]]

    ref_def = {
        "full": ref_p_full[index],
        "half": ref_p_half[index],
        "delta": ref_delta[index],
        "alpha": ref_alpha[index],
    }
    ref = {
        key: val
        for key, val in ref_def.items()
        if (output == key or (isinstance(output, (list, tuple)) and key in output))
    }

    levels = np.asarray(levels) if levels is not None else None

    if levels is not None:
        levels_half_idx = levels
        levels_idx = levels - 1
        for key in ref:
            if key == "half":
                ref[key] = ref[key][levels_half_idx]
            else:
                ref[key] = ref[key][levels_idx]

        # ref_p_full = ref_p_full[levels_idx, :]
        # ref_p_half = ref_p_half[levels_idx + 1, :]
        # ref_delta = ref_delta[levels_idx, :]
        # ref_alpha = ref_alpha[levels_idx, :]

    print("levels=", levels)

    res = vertical.pressure_on_hybrid_levels(A, B, sp, levels=levels, alpha_top="ifs", output=output)

    if isinstance(output, str) or len(output) == 1:
        key = output if isinstance(output, str) else output[0]
        assert xp.allclose(res, ref[key])
    else:
        assert isinstance(res, tuple)
        assert len(res) == len(output)
        for key, rd in zip(output, res):
            assert xp.allclose(rd, ref[key])

    # # print("p_full", repr(p_full))
    # # print("p_half", repr(p_half))
    # # print("delta", repr(delta))
    # # print("alpha", repr(alpha))

    # assert xp.allclose(p_full, ref_p_full)
    # assert xp.allclose(p_half, ref_p_half)
    # assert xp.allclose(delta, ref_delta)
    # assert xp.allclose(alpha, ref_alpha)
