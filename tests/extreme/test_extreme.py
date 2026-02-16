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
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo import extreme

here = os.path.dirname(__file__)
sys.path.insert(0, here)
import _cpf  # noqa
import _data  # noqa


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("clim,ens,v_ref", [(_data.clim, _data.ens, [-0.1838425040642013])])
def test_highlevel_efi(xp, device, clim, ens, v_ref):
    clim = xp.asarray(clim, device=device)
    ens = xp.asarray(ens, device=device)
    v_ref = xp.asarray(v_ref, device=device)
    # clim, ens, v_ref = array_backend.asarray(clim, ens, v_ref)
    efi = extreme.efi(clim, ens)
    assert xp.isclose(efi[0], v_ref[0])


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
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
def test_efi_core(xp, device, clim, ens, kwargs, v_ref):
    clim = xp.asarray(clim, device=device)
    ens = xp.asarray(ens, device=device)
    v_ref = xp.asarray(v_ref, device=device)
    efi = extreme.array.efi(clim, ens, **kwargs)
    assert xp.allclose(efi[0], v_ref, rtol=1e-4)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("clim,ens,v_ref", [(_data.clim, _data.ens, -0.18384250406420133)])
def test_efi_sorted(xp, device, clim, ens, v_ref):
    clim = xp.asarray(clim, device=device)
    ens = xp.asarray(ens, device=device)
    v_ref = xp.asarray(v_ref, device=device)

    # ensures the algorithm is the same if we sort the data or not
    ens_perc = xp.sort(ens)

    efi = extreme.array.efi(clim, ens_perc)

    assert xp.isclose(efi[0], v_ref)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_efi_nan(xp, device):
    clim_nan = xp.empty((101, 1), device=device)
    clim_nan[:] = xp.nan
    ens_nan = xp.empty((51, 1), device=device)
    ens_nan[:] = xp.nan
    # print(clim_nan)
    # print(ens_nan)

    efi = extreme.array.efi(clim_nan, ens_nan)

    # print(efi)
    assert xp.isnan(efi[0])


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("clim,ens,v_ref", [(_data.clim, _data.ens, [-2.14617638, -1.3086723])])
def test_sot_highlevel(xp, device, clim, ens, v_ref):
    clim = xp.asarray(clim, device=device)
    ens = xp.asarray(ens, device=device)
    v_ref = xp.asarray(v_ref, device=device)

    sot_upper = extreme.sot(clim, ens, 90)
    sot_lower = extreme.sot(clim, ens, 10)

    v_ref = xp.asarray(v_ref, dtype=sot_upper.dtype)

    assert xp.allclose(sot_upper[0], v_ref[0])
    assert xp.allclose(sot_lower[0], v_ref[1])


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "clim,ens,v_ref",
    [
        (_data.clim, _data.ens, [-2.14617638, -1.3086723]),
    ],
)
def test_sot_core(xp, device, clim, ens, v_ref):
    clim = xp.asarray(clim, device=device)
    ens = xp.asarray(ens, device=device)
    v_ref = xp.asarray(v_ref, device=device)

    sot_upper = extreme.array.sot(clim, ens, 90)
    sot_lower = extreme.array.sot(clim, ens, 10)

    v_ref = xp.asarray(v_ref, dtype=sot_upper.dtype)

    assert xp.allclose(sot_upper[0], v_ref[0])
    assert xp.allclose(sot_lower[0], v_ref[1])


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
# @pytest.mark.parametrize("array_backend", get_array_backend(["numpy"]))
@pytest.mark.parametrize("clim,ens,v_ref", [(_data.clim_eps2, _data.ens_eps2, [np.nan])])
def test_sot_perc(xp, device, clim, ens, v_ref):
    clim = xp.asarray(clim, device=device)
    ens = xp.asarray(ens, device=device)
    v_ref = xp.asarray(v_ref, device=device)

    sot = extreme.array.sot(clim, ens, 90, eps=1e4)

    v_ref = xp.asarray(v_ref, dtype=sot.dtype)

    assert xp.allclose(sot[0], v_ref[0], equal_nan=True)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
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
def test_sot_func(xp, device, qc_tail, qc, qf, kwargs, v_ref):
    qc_tail = xp.asarray(qc_tail, device=device)
    qc = xp.asarray(qc, device=device)
    qf = xp.asarray(qf, device=device)
    v_ref = xp.asarray(v_ref, device=device)

    sot = extreme.array.sot_func(qc_tail, qc, qf, **kwargs)

    v_ref = xp.asarray(v_ref, dtype=sot.dtype)

    assert xp.allclose(sot, v_ref, equal_nan=True)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize("clim,ens,v_ref", [(_cpf.cpf_clim, _cpf.cpf_ens, _cpf.cpf_val)])
def test_cpf_highlevel(xp, device, clim, ens, v_ref):
    clim = xp.asarray(clim, device=device)
    ens = xp.asarray(ens, device=device)
    v_ref = xp.asarray(v_ref, device=device)

    cpf = extreme.cpf(clim, ens, sort_clim=True)

    assert xp.allclose(cpf, v_ref)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "clim,ens,kwargs,v_ref",
    [
        (_cpf.cpf_clim, _cpf.cpf_ens, dict(sort_clim=True), _cpf.cpf_val),
        (_cpf.cpf_clim2, _cpf.cpf_ens2, dict(sort_clim=True, epsilon=0.5), _cpf.cpf_val2),  # eps
        (_cpf.cpf_clim3, _cpf.cpf_ens3, dict(sort_clim=True, symmetric=True), _cpf.cpf_val3),  # sym
        (_cpf.cpf_clim, _cpf.cpf_ens, dict(sort_clim=True, from_zero=True), _cpf.cpf_val_fromzero),
    ],
)
def test_cpf_core(xp, device, clim, ens, v_ref, kwargs):
    clim = xp.asarray(clim, device=device)
    ens = xp.asarray(ens, device=device)
    v_ref = xp.asarray(v_ref, device=device)

    cpf = extreme.array.cpf(clim, ens, **kwargs)
    assert xp.allclose(cpf, v_ref)
