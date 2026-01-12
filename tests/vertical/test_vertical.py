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

np.set_printoptions(formatter={"float_kind": "{:.15f}".format})


def _get_data(name):
    import os
    import sys

    here = os.path.dirname(__file__)
    sys.path.insert(0, here)

    return import_module(name)


DATA_HYBRID_CORE = _get_data("_hybrid_core_data")
DATA_HYBRID_H = _get_data("_hybrid_height_data")


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
def test_hybrid_level_parameters_1(xp):
    ref_A = DATA_HYBRID_CORE.A
    ref_B = DATA_HYBRID_CORE.B
    ref_A, ref_B = (xp.asarray(x) for x in [ref_A, ref_B])

    A, B = vertical.hybrid_level_parameters(137)

    # Note: A in test data has been stored with higher precision than in the conf
    assert np.allclose(A, ref_A, rtol=1e-5)
    assert np.allclose(B, ref_B, rtol=1e-5)


def test_hybrid_level_parameters_2():
    with pytest.raises(ValueError):
        vertical.hybrid_level_parameters(-100)


def test_hybrid_level_parameters_3():
    with pytest.raises(ValueError):
        vertical.hybrid_level_parameters(137, model="unknown_model")


@pytest.mark.parametrize("xp", [_NUMPY_NAMESPACE])
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
def test_pressure_on_hybrid_levels_1(index, xp):

    sp = DATA_HYBRID_CORE.p_surf
    A = DATA_HYBRID_CORE.A
    B = DATA_HYBRID_CORE.B
    ref_p_full = DATA_HYBRID_CORE.p_full
    ref_p_half = DATA_HYBRID_CORE.p_half
    ref_delta = DATA_HYBRID_CORE.delta
    ref_alpha = DATA_HYBRID_CORE.alpha

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

    sp = DATA_HYBRID_CORE.p_surf
    A = DATA_HYBRID_CORE.A
    B = DATA_HYBRID_CORE.B
    ref_p_full = DATA_HYBRID_CORE.p_full
    ref_p_half = DATA_HYBRID_CORE.p_half
    ref_delta = DATA_HYBRID_CORE.delta
    ref_alpha = DATA_HYBRID_CORE.alpha

    # ref_def = {"full": DATA.p_full, "half": DATA.p_half, "delta": DATA_HYBRID_CORE.delta, "alpha": DATA_HYBRID_CORE.alpha}
    # ref = {
    #     key: val
    #     for key, val in ref_def.items()
    #     if (output == key or (isinstance(output, (list, tuple)) and key in output))
    # }

    # ref_p_full = DATA_HYBRID_CORE.p_full
    # ref_p_half = DATA_HYBRID_CORE.p_half
    # ref_delta = DATA_HYBRID_CORE.delta
    # ref_alpha = DATA_HYBRID_CORE.alpha

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


# @pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("xp, device", [(_NUMPY_NAMESPACE, "cpu")])
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
def test_relative_geopotential_thickness_on_hybrid_levels_1(index, xp, device):

    alpha = DATA_HYBRID_CORE.alpha
    delta = DATA_HYBRID_CORE.delta
    t = DATA_HYBRID_CORE.t
    q = DATA_HYBRID_CORE.q
    z_ref = DATA_HYBRID_CORE.z

    z_ref, t, q, alpha, delta = (xp.asarray(x, device=device) for x in [z_ref, t, q, alpha, delta])

    alpha = alpha[index]
    delta = delta[index]
    t = t[index]
    q = q[index]
    z_ref = z_ref[index]

    z = vertical.relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta(t, q, alpha, delta)

    assert xp.allclose(z, z_ref)


# @pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("xp, device", [(_NUMPY_NAMESPACE, "cpu")])
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
def test_relative_geopotential_thickness_on_hybrid_levels_2(index, xp, device):

    A = DATA_HYBRID_CORE.A
    B = DATA_HYBRID_CORE.B
    sp = DATA_HYBRID_CORE.p_surf
    t = DATA_HYBRID_CORE.t
    q = DATA_HYBRID_CORE.q
    z_ref = DATA_HYBRID_CORE.z

    z_ref, t, q, A, B, sp = (xp.asarray(x, device=device) for x in [z_ref, t, q, A, B, sp])

    sp = sp[index[1]]
    t = t[index]
    q = q[index]
    z_ref = z_ref[index]

    z = vertical.relative_geopotential_thickness_on_hybrid_levels(t, q, A, B, sp)

    assert xp.allclose(z, z_ref, rtol=1e-6)


# @pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("xp, device", [(_NUMPY_NAMESPACE, "cpu")])
@pytest.mark.parametrize(
    "index",
    [
        (slice(None), slice(None)),
    ],
)
# @pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
def test_relative_geopotential_thickness_on_hybrid_levels_part(index, xp, device):
    # get only levels from 90 to 136/137
    part = slice(90, None)

    A = DATA_HYBRID_CORE.A
    B = DATA_HYBRID_CORE.B
    sp = DATA_HYBRID_CORE.p_surf
    t = DATA_HYBRID_CORE.t
    q = DATA_HYBRID_CORE.q
    z_ref = DATA_HYBRID_CORE.z

    z_ref, t, q, A, B, sp = (xp.asarray(x, device=device) for x in [z_ref, t, q, A, B, sp])

    part_index = (part, index[1])

    sp = sp[index[1]]
    t = t[part_index]
    q = q[part_index]
    z_ref = z_ref[part_index]

    z = vertical.relative_geopotential_thickness_on_hybrid_levels(t, q, A, B, sp)

    assert xp.allclose(z, z_ref, rtol=1e-6)


# @pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("xp, device", [(_NUMPY_NAMESPACE, "cpu")])
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
def test_geopotential_on_hybrid_levels(index, xp, device):

    A = DATA_HYBRID_CORE.A
    B = DATA_HYBRID_CORE.B
    sp = DATA_HYBRID_CORE.p_surf
    t = DATA_HYBRID_CORE.t
    q = DATA_HYBRID_CORE.q
    z_ref = DATA_HYBRID_CORE.z
    zs = [0.0] * len(t[0])  # surface geopotential is zero in test data

    z_ref, t, q, zs, A, B, sp = (xp.asarray(x, device=device) for x in [z_ref, t, q, zs, A, B, sp])

    sp = sp[index[1]]
    t = t[index]
    q = q[index]
    z_ref = z_ref[index]
    zs = zs[index[1]]

    z = vertical.geopotential_on_hybrid_levels(t, q, zs, A, B, sp)

    assert xp.allclose(z, z_ref, rtol=1e-6)


@pytest.mark.parametrize("xp, device", [(_NUMPY_NAMESPACE, "cpu")])
@pytest.mark.parametrize("index", [(slice(None), slice(None)), (slice(None), 0), (slice(None), 1)])
@pytest.mark.parametrize("h_type", ["geometric", "geopotential"])
@pytest.mark.parametrize("h_reference", ["sea", "ground"])
def test_height_on_hybrid_levels(index, xp, device, h_type, h_reference):

    A, B = vertical.hybrid_level_parameters(137)
    sp = DATA_HYBRID_H.p_surf
    t = DATA_HYBRID_H.t
    q = DATA_HYBRID_H.q
    zs = DATA_HYBRID_H.z_surf

    ref_name = f"h_{h_type}_{h_reference}"
    h_ref = getattr(DATA_HYBRID_H, ref_name)

    h_ref, t, q, zs, sp, A, B = (xp.asarray(x, device=device) for x in [h_ref, t, q, zs, sp, A, B])

    t = t[index]
    q = q[index]
    h_ref = h_ref[index]
    zs = zs[index[1]]
    sp = sp[index[1]]

    h = vertical.height_on_hybrid_levels(t, q, zs, A, B, sp, h_type=h_type, h_reference=h_reference)
    assert xp.allclose(h, h_ref)
