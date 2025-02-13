# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import os
import sys

import numpy as np
import pytest

from earthkit.meteo import extreme
from earthkit.meteo.utils.testing import ARRAY_BACKENDS

here = os.path.dirname(__file__)
sys.path.insert(0, here)
import _cpf  # noqa
import _data  # noqa


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("clim,ens,v_ref", [(_data.clim, _data.ens, [-0.1838425040642013])])
def test_highlevel_efi(clim, ens, v_ref, array_backend):
    clim, ens, v_ref = array_backend.asarray(clim, ens, v_ref)
    efi = extreme.efi(clim, ens)
    assert array_backend.isclose(efi[0], v_ref[0])


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "clim,ens,kwargs,v_ref",
    [
        (_data.clim, _data.ens, {}, -0.1838425040642013),
        (_data.clim, _data.ens, dict(eps=1e-4), -0.18384250406420133),
        (
            _data.clim_eps,
            _data.ens_eps,
            dict(eps=1e-4),
            0.46039347745967046,
        ),  # fortran code result is  0.4604220986366272
        (_data.clim_eps2, _data.ens_eps2, dict(eps=1e-4), 0.6330071575726789),
    ],
)
def test_efi_core(clim, ens, kwargs, v_ref, array_backend):
    clim, ens, v_ref = array_backend.asarray(clim, ens, v_ref)
    efi = extreme.array.efi(clim, ens, **kwargs)
    assert array_backend.allclose(efi[0], v_ref, rtol=1e-4)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("clim,ens,v_ref", [(_data.clim, _data.ens, -0.18384250406420133)])
def test_efi_sorted(clim, ens, v_ref, array_backend):
    clim, ens, v_ref = array_backend.asarray(clim, ens, v_ref)

    # ensures the algorithm is the same if we sort the data or not
    ens_perc = array_backend.namespace.sort(ens)

    efi = extreme.array.efi(clim, ens_perc)

    assert array_backend.isclose(efi[0], v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
def test_efi_nan(array_backend):
    xp = array_backend.namespace

    clim_nan = xp.empty((101, 1))
    clim_nan[:] = xp.nan
    ens_nan = xp.empty((51, 1))
    ens_nan[:] = xp.nan
    # print(clim_nan)
    # print(ens_nan)

    efi = extreme.array.efi(clim_nan, ens_nan)

    # print(efi)
    assert xp.isnan(efi[0])


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("clim,ens,v_ref", [(_data.clim, _data.ens, [-2.14617638, -1.3086723])])
def test_sot_highlevel(clim, ens, v_ref, array_backend):
    clim, ens, v_ref = array_backend.asarray(clim, ens, v_ref)

    sot_upper = extreme.sot(clim, ens, 90)
    sot_lower = extreme.sot(clim, ens, 10)

    v_ref = array_backend.asarray(v_ref, dtype=sot_upper.dtype)

    assert array_backend.allclose(sot_upper[0], v_ref[0])
    assert array_backend.allclose(sot_lower[0], v_ref[1])


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
# @pytest.mark.parametrize("array_backend", get_array_backend(["numpy"]))
@pytest.mark.parametrize(
    "clim,ens,v_ref",
    [
        (_data.clim, _data.ens, [-2.14617638, -1.3086723]),
    ],
)
def test_sot_core(clim, ens, v_ref, array_backend):
    clim, ens, v_ref = array_backend.asarray(clim, ens, v_ref)

    sot_upper = extreme.array.sot(clim, ens, 90)
    sot_lower = extreme.array.sot(clim, ens, 10)

    v_ref = array_backend.asarray(v_ref, dtype=sot_upper.dtype)

    # print(sot_upper)
    # print(sot_lower)

    assert array_backend.allclose(sot_upper[0], v_ref[0])
    assert array_backend.allclose(sot_lower[0], v_ref[1])


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
# @pytest.mark.parametrize("array_backend", get_array_backend(["numpy"]))
@pytest.mark.parametrize("clim,ens,v_ref", [(_data.clim_eps2, _data.ens_eps2, [np.nan])])
def test_sot_perc(clim, ens, v_ref, array_backend):
    clim, ens, v_ref = array_backend.asarray(clim, ens, v_ref)

    sot = extreme.array.sot(clim, ens, 90, eps=1e4)

    v_ref = array_backend.asarray(v_ref, dtype=sot.dtype)

    assert array_backend.allclose(sot[0], v_ref[0], equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
# @pytest.mark.parametrize("array_backend", get_array_backend(["numpy"]))
@pytest.mark.parametrize(
    "qc_tail,qc,qf,kwargs,v_ref",
    [
        (
            [1.0, 1.0, 1.0, 1.0],
            [1.1, 1.0, 1.0, 1.00001],
            [1.5, 1.2, 1.0, 0.9],
            dict(eps=1e-4),
            [-5.0, np.nan, np.nan, np.nan],
        ),  # first value valid, second -> inf, third gives nan, third is below threshold -> nan
        ([1.0, 1.0], [1.1, 1.1], [15, -15.0], {}, [-10, 10]),  # bounds
        (
            [0.05],
            [0.1],
            [0.2],
            {},
            [-3.0],
        ),  # eps:  first value valid, second -> inf, third gives nan, third is below threshold -> nan
        (
            [0.05],
            [0.1],
            [0.2],
            dict(eps=0.15),
            [np.nan],
        ),  # eps:  first value valid, second -> inf, third gives nan, third is below threshold -> nan
        ([0.05], [0.1], [np.nan], {}, [np.nan]),  # nan
        ([0.05], [np.nan], [0.1], {}, [np.nan]),  # nan
        ([np.nan], [0.1], [0.2], {}, [np.nan]),  # nan
    ],
)
def test_sot_func(qc_tail, qc, qf, kwargs, v_ref, array_backend):
    qc_tail, qc, qf, v_ref = array_backend.asarray(qc_tail, qc, qf, v_ref)

    sot = extreme.array.sot_func(qc_tail, qc, qf, **kwargs)

    v_ref = array_backend.asarray(v_ref, dtype=sot.dtype)

    assert array_backend.allclose(sot, v_ref, equal_nan=True)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("clim,ens,v_ref", [(_cpf.cpf_clim, _cpf.cpf_ens, _cpf.cpf_val)])
def test_cpf_highlevel(clim, ens, v_ref, array_backend):
    clim, ens, v_ref = array_backend.asarray(clim, ens, v_ref)

    cpf = extreme.cpf(clim, ens, sort_clim=True)

    assert array_backend.allclose(cpf, v_ref)


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "clim,ens,kwargs,v_ref",
    [
        (_cpf.cpf_clim, _cpf.cpf_ens, dict(sort_clim=True), _cpf.cpf_val),
        (_cpf.cpf_clim2, _cpf.cpf_ens2, dict(sort_clim=True, epsilon=0.5), _cpf.cpf_val2),  # eps
        # (_cpf.cpf_clim3, _cpf.cpf_ens3, dict(sort_clim=True, symmetric=True), _cpf.cpf_val3),  # sym
    ],
)
def test_cpf_core(clim, ens, v_ref, kwargs, array_backend):
    clim, ens, v_ref = array_backend.asarray(clim, ens, v_ref)

    cpf = extreme.array.cpf(clim, ens, **kwargs)
    assert array_backend.allclose(cpf, v_ref)
