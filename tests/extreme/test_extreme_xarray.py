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
import time

import numpy as np
import pytest

from earthkit.meteo import extreme
from earthkit.meteo.extreme import xarray as extreme_xr
from earthkit.meteo.utils.testing import NO_XARRAY

here = os.path.dirname(__file__)
sys.path.insert(0, here)
import _cpf  # noqa
import _data  # noqa

pytestmark = pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")


def _da(values, dims):
    import xarray as xr

    data = np.asarray(values)
    return xr.DataArray(data, dims=dims)


def _da_move_dim(values, dims, axis, axis_orig=0):
    data = _da(values, dims)
    new_dims = list(dims)
    pop_dim = new_dims.pop(axis_orig)
    new_dims.insert(axis, pop_dim)
    return data.transpose(*new_dims)


# ---------------------------------
# EFI
# ---------------------------------


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_efi():
    clim = _da(_data.clim, dims=["quantiles", "values", "x", "y"])
    ens = _da(_data.ens, dims=["number", "values", "x", "y"])
    v_ref = -0.1838425040642013
    efi = extreme_xr.efi(clim, ens)
    assert np.isclose(efi.values.flat[0], v_ref)


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_xr_efi_dims(axis):
    clim = _da_move_dim(_data.clim, dims=["quantiles", "values", "x", "y"], axis=axis)
    ens = _da_move_dim(_data.ens, dims=["number", "values", "x", "y"], axis=axis)
    v_ref = -0.1838425040642013
    efi = extreme_xr.efi(clim, ens)
    assert np.isclose(efi.values.flat[0], v_ref)


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_efi_mixed_dims():
    clim = _da(_data.clim, dims=["quantiles", "values", "x", "y"])
    ens = _da_move_dim(_data.ens, dims=["number", "values", "x", "y"], axis=-1)
    v_ref = -0.1838425040642013
    efi = extreme.efi(clim, ens, clim_dim="quantiles", ens_dim="number")
    assert np.isclose(efi.values.flat[0], v_ref)


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_efi_mixed_dims_even_more():
    clim = _da(_data.clim, dims=["quantiles", "values", "x", "y"])
    ens = _da_move_dim(_data.ens, dims=["number", "values", "x", "y"], axis=-1, axis_orig=1)
    v_ref = -0.1838425040642013
    efi = extreme.efi(clim, ens, clim_dim="quantiles", ens_dim="number")
    assert np.isclose(efi.values.flat[0], v_ref)


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_efi_highlevel():
    clim = _da(_data.clim, dims=["quantiles", "values", "x", "y"])
    ens = _da(_data.ens, dims=["number", "values", "x", "y"])
    v_ref = -0.1838425040642013
    efi = extreme.efi(clim, ens)
    assert np.isclose(efi.values.flat[0], v_ref)


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_efi_perf():
    clim = _data.clim
    ens = _data.ens
    clim = np.repeat(clim, 10, axis=-2)
    clim = np.repeat(clim, 10, axis=-1)
    ens = np.repeat(ens, 10, axis=-2)
    ens = np.repeat(ens, 10, axis=-1)
    clim_da = _da(clim, dims=("quantiles", "values", "x", "y"))
    ens_da = _da(ens, dims=("number", "values", "x", "y"))

    # Warm up to reduce first-call overhead in timing.
    extreme.array.efi(clim, ens)
    extreme_xr.efi(clim_da, ens_da)
    extreme.efi(clim_da, ens_da)

    repeats = 5
    t0 = time.perf_counter()
    for _ in range(repeats):
        extreme.array.efi(clim, ens)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(repeats):
        extreme_xr.efi(clim_da, ens_da)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    for _ in range(repeats):
        extreme.efi(clim_da, ens_da)
    t5 = time.perf_counter()

    array_time = (t1 - t0) / repeats
    xarray_time = (t3 - t2) / repeats
    highlevel_time = (t5 - t4) / repeats
    print(f"Array EFI time: {array_time:.6f} s")
    print(f"xarray EFI time: {xarray_time:.6f} s")
    print(f"High-level EFI time: {highlevel_time:.6f} s")
    assert xarray_time <= 1.5 * array_time
    assert highlevel_time <= 1.5 * array_time


# ---------------------------------
# SOT
# ---------------------------------


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_sot():
    clim = _da(_data.clim, dims=("quantiles", "values", "x", "y"))
    ens = _da(_data.ens, dims=("number", "values", "x", "y"))
    v_ref = [-2.14617638, -1.3086723]
    sot_upper = extreme_xr.sot(clim, ens, 90)
    sot_lower = extreme_xr.sot(clim, ens, 10)
    assert np.allclose(sot_upper.values.flat[0], v_ref[0])
    assert np.allclose(sot_lower.values.flat[0], v_ref[1])


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_xr_sot_dims(axis):
    clim = _da_move_dim(_data.clim, dims=("quantiles", "values", "x", "y"), axis=axis)
    ens = _da_move_dim(_data.ens, dims=("number", "values", "x", "y"), axis=axis)
    v_ref = [-2.14617638, -1.3086723]
    sot_upper = extreme.sot(clim, ens, 90)
    sot_lower = extreme.sot(clim, ens, 10)
    assert np.allclose(sot_upper.values.flat[0], v_ref[0])
    assert np.allclose(sot_lower.values.flat[0], v_ref[1])


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_sot_mixed_dims():
    clim = _da(_data.clim, dims=("quantiles", "values", "x", "y"))
    ens = _da_move_dim(_data.ens, dims=("number", "values", "x", "y"), axis=-1)
    v_ref = [-2.14617638, -1.3086723]
    sot_upper = extreme.sot(clim, ens, 90)
    sot_lower = extreme.sot(clim, ens, 10)
    assert np.allclose(sot_upper.values.flat[0], v_ref[0])
    assert np.allclose(sot_lower.values.flat[0], v_ref[1])


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_sot_highlevel():
    clim = _da(_data.clim, dims=("quantiles", "values", "x", "y"))
    ens = _da(_data.ens, dims=("number", "values", "x", "y"))
    v_ref = [-2.14617638, -1.3086723]
    sot_upper = extreme.sot(clim, ens, 90)
    sot_lower = extreme.sot(clim, ens, 10)
    assert np.allclose(sot_upper.values[0], v_ref[0])
    assert np.allclose(sot_lower.values[0], v_ref[1])


# ---------------------------------
# CPF
# ---------------------------------


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_cpf():
    clim = _da(_cpf.cpf_clim, dims=("quantiles", "values", "x", "y"))
    ens = _da(_cpf.cpf_ens, dims=("number", "values", "x", "y"))
    v_ref = _cpf.cpf_val
    cpf = extreme_xr.cpf(clim, ens, sort_clim=True)
    assert np.allclose(cpf.values.flat, np.asarray(v_ref))


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_xr_cpf_dims(axis):
    clim = _da_move_dim(_cpf.cpf_clim, dims=("quantiles", "values", "x", "y"), axis=axis)
    ens = _da_move_dim(_cpf.cpf_ens, dims=("number", "values", "x", "y"), axis=axis)
    v_ref = _cpf.cpf_val
    cpf = extreme.cpf(clim, ens, sort_clim=True)
    assert np.allclose(cpf.values.flat, np.asarray(v_ref))


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_cpf_mixed_dims():
    clim = _da(_cpf.cpf_clim, dims=("quantiles", "values", "x", "y"))
    ens = _da_move_dim(_cpf.cpf_ens, dims=("number", "values", "x", "y"), axis=-1)
    v_ref = _cpf.cpf_val
    cpf = extreme.cpf(clim, ens, sort_clim=True)
    assert np.allclose(cpf.values.flat, np.asarray(v_ref))


@pytest.mark.skipif(NO_XARRAY, reason="xarray is not installed")
def test_xr_cpf_highlevel():
    clim = _da(_cpf.cpf_clim, dims=("quantiles", "values", "x", "y"))
    ens = _da(_cpf.cpf_ens, dims=("number", "values", "x", "y"))
    v_ref = _cpf.cpf_val
    cpf = extreme.cpf(
        _da(clim, dims=("quantiles", "values", "x", "y")),
        _da(ens, dims=("number", "values", "x", "y")),
        sort_clim=True,
    )
    assert np.allclose(cpf.values.flat, np.asarray(v_ref))
