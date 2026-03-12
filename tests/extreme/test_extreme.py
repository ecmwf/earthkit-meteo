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

from earthkit.meteo import extreme

from . import _cpf
from . import _data


def _move_axis(data, axis):
    return np.moveaxis(np.asarray(data), 0, axis)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
@pytest.mark.parametrize(
    "clim, ens, kwargs, v_ref",
    [
        (_data.clim, _data.ens, {}, -0.1838425040642013),
        (_data.clim, _data.ens, dict(eps=1e-4), -0.18384250406420133),
        (_data.clim_eps, _data.ens_eps, dict(eps=1e-4), 0.46039347745967046),
        (_data.clim_eps2, _data.ens_eps2, dict(eps=1e-4), 0.6330071575726789),
    ],
)
def test_np_efi_highlevel_dispatch(xp, device, clim, ens, kwargs, v_ref):
    clim = xp.asarray(clim, device=device)
    ens = xp.asarray(ens, device=device)
    v_ref = xp.asarray(v_ref, device=device)
    efi = extreme.efi(clim, ens, **kwargs)
    assert xp.allclose(efi[0], v_ref, rtol=1e-4)


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_np_efi_highlevel_dispatch_axis(axis):
    clim = _move_axis(_data.clim, axis)
    ens = _move_axis(_data.ens, axis)
    ref = extreme.array.efi(_data.clim, _data.ens)
    got = extreme.efi(clim, ens, clim_dim=axis, ens_dim=axis)
    assert np.allclose(got, ref)


@pytest.mark.parametrize("axis", [1, 2, 3])
def test_np_efi_highlevel_dispatch_mixed_axis(axis):
    clim = _data.clim
    ens = _move_axis(_data.ens, axis)
    ref = extreme.array.efi(_data.clim, _data.ens)
    got = extreme.efi(clim, ens, clim_dim=0, ens_dim=axis)
    assert np.allclose(got, ref)


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_np_sot_highlevel_dispatch_axis(axis):
    clim = _move_axis(_data.clim, axis)
    ens = _move_axis(_data.ens, axis)
    ref = extreme.array.sot(_data.clim, _data.ens, 90)
    got = extreme.sot(clim, ens, 90, clim_dim=axis, ens_dim=axis)
    assert np.allclose(got, ref)


@pytest.mark.parametrize("axis", [1, 2, 3])
def test_np_sot_highlevel_dispatch_mixed_axis(axis):
    clim = _data.clim
    ens = _move_axis(_data.ens, axis)
    ref = extreme.array.sot(_data.clim, _data.ens, 90)
    got = extreme.sot(clim, ens, 90, clim_dim=0, ens_dim=axis)
    assert np.allclose(got, ref)


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_np_cpf_highlevel_dispatch_axis(axis):
    clim = _move_axis(_cpf.cpf_clim, axis)
    ens = _move_axis(_cpf.cpf_ens, axis)
    ref = extreme.array.cpf(_cpf.cpf_clim, _cpf.cpf_ens, sort_clim=True)
    got = extreme.cpf(clim, ens, sort_clim=True, clim_dim=axis, ens_dim=axis)
    assert np.allclose(got, ref)


@pytest.mark.parametrize("axis", [1, 2, 3])
def test_np_cpf_highlevel_dispatch_mixed_axis(axis):
    clim = _cpf.cpf_clim
    ens = _move_axis(_cpf.cpf_ens, axis)
    ref = extreme.array.cpf(_cpf.cpf_clim, _cpf.cpf_ens, sort_clim=True)
    got = extreme.cpf(clim, ens, sort_clim=True, clim_dim=0, ens_dim=axis)
    assert np.allclose(got, ref)


xr = pytest.importorskip("xarray")


def _da(values, dims):
    data = np.asarray(values)
    return xr.DataArray(data, dims=dims)


def _da_move_dim(values, dims, axis, axis_orig=0):
    data = _da(values, dims)
    new_dims = list(dims)
    pop_dim = new_dims.pop(axis_orig)
    new_dims.insert(axis, pop_dim)
    return data.transpose(*new_dims)


def test_xr_efi_mixed_dims():
    clim = _da(_data.clim, dims=["quantiles", "values", "x", "y"])
    ens = _da_move_dim(_data.ens, dims=["number", "values", "x", "y"], axis=-1)
    v_ref = -0.1838425040642013
    efi = extreme.efi(clim, ens, clim_dim="quantiles", ens_dim="number")
    assert np.isclose(efi.values.flat[0], v_ref)


def test_xr_efi_mixed_dims_even_more():
    clim = _da(_data.clim, dims=["quantiles", "values", "x", "y"])
    ens = _da_move_dim(_data.ens, dims=["number", "values", "x", "y"], axis=-1, axis_orig=1)
    v_ref = -0.1838425040642013
    efi = extreme.efi(clim, ens, clim_dim="quantiles", ens_dim="number")
    assert np.isclose(efi.values.flat[0], v_ref)


def test_xr_efi_highlevel():
    clim = _da(_data.clim, dims=["quantiles", "values", "x", "y"])
    ens = _da(_data.ens, dims=["number", "values", "x", "y"])
    v_ref = -0.1838425040642013
    efi = extreme.efi(clim, ens)
    assert np.isclose(efi.values.flat[0], v_ref)


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_xr_sot_dims(axis):
    clim = _da_move_dim(_data.clim, dims=("quantiles", "values", "x", "y"), axis=axis)
    ens = _da_move_dim(_data.ens, dims=("number", "values", "x", "y"), axis=axis)
    v_ref = [-2.14617638, -1.3086723]
    sot_upper = extreme.sot(clim, ens, 90)
    sot_lower = extreme.sot(clim, ens, 10)
    assert np.allclose(sot_upper.values.flat[0], v_ref[0])
    assert np.allclose(sot_lower.values.flat[0], v_ref[1])


def test_xr_sot_mixed_dims():
    clim = _da(_data.clim, dims=("quantiles", "values", "x", "y"))
    ens = _da_move_dim(_data.ens, dims=("number", "values", "x", "y"), axis=-1)
    v_ref = [-2.14617638, -1.3086723]
    sot_upper = extreme.sot(clim, ens, 90)
    sot_lower = extreme.sot(clim, ens, 10)
    assert np.allclose(sot_upper.values.flat[0], v_ref[0])
    assert np.allclose(sot_lower.values.flat[0], v_ref[1])


def test_xr_sot_highlevel():
    clim = _da(_data.clim, dims=("quantiles", "values", "x", "y"))
    ens = _da(_data.ens, dims=("number", "values", "x", "y"))
    v_ref = [-2.14617638, -1.3086723]
    sot_upper = extreme.sot(clim, ens, 90)
    sot_lower = extreme.sot(clim, ens, 10)
    assert np.allclose(sot_upper.values[0], v_ref[0])
    assert np.allclose(sot_lower.values[0], v_ref[1])


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_xr_cpf_dims(axis):
    clim = _da_move_dim(_cpf.cpf_clim, dims=("quantiles", "values", "x", "y"), axis=axis)
    ens = _da_move_dim(_cpf.cpf_ens, dims=("number", "values", "x", "y"), axis=axis)
    v_ref = _cpf.cpf_val
    cpf = extreme.cpf(clim, ens, sort_clim=True)
    assert np.allclose(cpf.values.flat, np.asarray(v_ref))


def test_xr_cpf_mixed_dims():
    clim = _da(_cpf.cpf_clim, dims=("quantiles", "values", "x", "y"))
    ens = _da_move_dim(_cpf.cpf_ens, dims=("number", "values", "x", "y"), axis=-1)
    v_ref = _cpf.cpf_val
    cpf = extreme.cpf(clim, ens, sort_clim=True)
    assert np.allclose(cpf.values.flat, np.asarray(v_ref))


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
