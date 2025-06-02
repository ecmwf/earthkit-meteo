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
from earthkit.meteo.utils.testing import ARRAY_BACKENDS
from earthkit.meteo.utils.testing import NUMPY_BACKEND

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


def _get_data():
    import os
    import sys

    here = os.path.dirname(__file__)
    sys.path.insert(0, here)
    import _vertical_data

    return _vertical_data


DATA = _get_data()


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "z,expected_value",
    [(0.0, 0.0), (1000.0, 101.97162129779284), ([1000.0, 10000.0], [101.9716212978, 1019.7162129779])],
)
def test_geopotential_height_from_geopotential(z, expected_value, array_backend):
    z, expected_value = array_backend.asarray(z, expected_value)

    r = vertical.geopotential_height_from_geopotential(z)
    assert array_backend.allclose(r, expected_value)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "h,expected_value",
    [
        (0.0, 0.0),
        (5102.664476187331, 50000.0),
        ([1019.8794448450, 5102.6644761873, 7146.0195417809], [10000.0, 50000.0, 70000.0]),
    ],
)
def test_geopotential_from_geometric_height(h, expected_value, array_backend):
    h, expected_value = array_backend.asarray(h, expected_value)

    r = vertical.geopotential_from_geometric_height(h)
    assert array_backend.allclose(r, expected_value)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "h,expected_value",
    [
        (0.0, 0.0),
        (5003.9269715243, 5000.0),
        ([1000.1569802279, 5003.9269715243, 7007.6992829768], [1000.0, 5000.0, 7000.0]),
    ],
)
def test_geopotential_height_from_geometric_height(h, expected_value, array_backend):
    h, expected_value = array_backend.asarray(h, expected_value)

    r = vertical.geopotential_height_from_geometric_height(h)

    assert array_backend.allclose(r, expected_value)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "z,expected_value",
    [
        (0.0, 0.0),
        (50000.0, 5102.664476187331),
        ([10000.0, 50000.0, 70000.0], [1019.8794448450, 5102.6644761873, 7146.0195417809]),
    ],
)
def test_geometric_height_from_geopotential(z, expected_value, array_backend):
    z, expected_value = array_backend.asarray(z, expected_value)

    r = vertical.geometric_height_from_geopotential(z)
    assert array_backend.allclose(r, expected_value)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "zh,expected_value",
    [
        (0.0, 0.0),
        (5000.0, 5003.9269715243),
        ([1000.0, 5000.0, 7000.0], [1000.1569802279, 5003.9269715243, 7007.6992829768]),
    ],
)
def test_geometric_height_from_geopotential_height(zh, expected_value, array_backend):
    zh, expected_value = array_backend.asarray(zh, expected_value)

    r = vertical.geometric_height_from_geopotential_height(zh)
    assert array_backend.allclose(r, expected_value)


@pytest.mark.parametrize("array_backend", [NUMPY_BACKEND])
def test_pressure_at_model_levels(array_backend):
    sp = 100000.0  # surface pressure in Pa

    A = DATA.A
    B = DATA.B
    ref_p_full = DATA.p_full
    ref_p_half = DATA.p_half
    ref_delta = DATA.delta
    ref_alpha = DATA.alpha

    sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B = array_backend.asarray(
        sp, ref_p_full, ref_p_half, ref_delta, ref_alpha, A, B
    )

    p_full, p_half, delta, alpha = vertical.pressure_at_model_levels(A, B, sp)

    assert array_backend.allclose(p_full, ref_p_full)
    assert array_backend.allclose(p_half, ref_p_half)
    assert array_backend.allclose(delta, ref_delta)
    assert array_backend.allclose(alpha, ref_alpha)


@pytest.mark.parametrize("array_backend", [NUMPY_BACKEND])
def test_relative_geopotential_thickness(array_backend):

    A = DATA.A
    B = DATA.B
    alpha = DATA.alpha
    t = DATA.t
    q = DATA.q
    z_ref = DATA.z

    z_ref, t, q, alpha, A, B = array_backend.asarray(z_ref, t, q, alpha, A, B)

    z = vertical.relative_geopotential_thickness(alpha, q, t)

    assert array_backend.allclose(z, z_ref)


# @pytest.mark.skipif(True, reason="Method needs to be fixed")
@pytest.mark.parametrize("array_backend", [NUMPY_BACKEND])
def test_pressure_at_height_level(array_backend):
    sp = 100000.0  # surface pressure in Pa
    h = 5000.0  # height in meters above surface
    A = DATA.A
    B = DATA.B
    t = DATA.t
    q = DATA.q

    sp, h, t, q, A, B = array_backend.asarray(sp, h, t, q, A, B)

    # xp = array_backend.namespace
    # t = xp.flip(t)
    # q = xp.flip(q)

    p = vertical.pressure_at_height_level(h, q, t, sp, A, B)
    print("p", p)
