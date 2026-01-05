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

from earthkit.meteo import vertical

np.set_printoptions(formatter={"float_kind": "{:.15f}".format})


# These are tests for deprecated functions in vertical


def _get_data(name):
    import os
    import sys

    here = os.path.dirname(__file__)
    sys.path.insert(0, here)

    return import_module(name)


DATA = _get_data("_hybrid_core_data")


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

    assert xp.allclose(p_full, ref_p_full, rtol=1e-8)
    assert xp.allclose(p_half, ref_p_half, rtol=1e-8)
    assert xp.allclose(delta, ref_delta, rtol=1e-8)
    assert xp.allclose(alpha, ref_alpha, rtol=1e-8)


# @pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("xp, device", [(_NUMPY_NAMESPACE, "cpu")])
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

    assert xp.allclose(z, z_ref, rtol=1e-8)


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
