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

xr = pytest.importorskip("xarray")

from earthkit.meteo.extreme import xarray as extreme_xr

from .. import _cpf
from .. import _data


def _da(values, dims):
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


def test_xr_efi():
    clim = _da(_data.clim, dims=["quantiles", "values", "x", "y"])
    ens = _da(_data.ens, dims=["number", "values", "x", "y"])
    v_ref = -0.1838425040642013
    efi = extreme_xr.efi(clim, ens)
    assert np.isclose(efi.values.flat[0], v_ref)


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_xr_efi_dims(axis):
    clim = _da_move_dim(_data.clim, dims=["quantiles", "values", "x", "y"], axis=axis)
    ens = _da_move_dim(_data.ens, dims=["number", "values", "x", "y"], axis=axis)
    v_ref = -0.1838425040642013
    efi = extreme_xr.efi(clim, ens)
    assert np.isclose(efi.values.flat[0], v_ref)


# ---------------------------------
# SOT
# ---------------------------------


def test_xr_sot():
    clim = _da(_data.clim, dims=("quantiles", "values", "x", "y"))
    ens = _da(_data.ens, dims=("number", "values", "x", "y"))
    v_ref = [-2.14617638, -1.3086723]
    sot_upper = extreme_xr.sot(clim, ens, 90)
    sot_lower = extreme_xr.sot(clim, ens, 10)
    assert np.allclose(sot_upper.values.flat[0], v_ref[0])
    assert np.allclose(sot_lower.values.flat[0], v_ref[1])


# ---------------------------------
# CPF
# ---------------------------------


def test_xr_cpf():
    clim = _da(_cpf.cpf_clim, dims=("quantiles", "values", "x", "y"))
    ens = _da(_cpf.cpf_ens, dims=("number", "values", "x", "y"))
    v_ref = _cpf.cpf_val
    cpf = extreme_xr.cpf(clim, ens, sort_clim=True)
    assert np.allclose(cpf.values.flat, np.asarray(v_ref))
