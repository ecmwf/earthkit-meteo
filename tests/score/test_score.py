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

from earthkit.meteo import score
from earthkit.meteo.utils.testing import ARRAY_BACKENDS
from earthkit.meteo.utils.testing import get_array_backend


def crps_quaver2(x, y):
    """Compute Continuous Ranked Probability Score (CRPS) from Quaver
    Used for testing

    Parameters
    ----------
    x: ndarray (n_ens, n_points)
        Ensemble forecast
    y: ndarray (n_points)
        Observation/analysis

    Returns
    -------
    ndarray (n_points)
        CRPS values

    The method is described in [Hersbach2000]_.
    """

    n_ens = x.shape[0]
    anarr = y
    earr = x
    # ensemble sorted by fieldset axis
    esarr = np.sort(earr, axis=0)
    aa = np.zeros(earr.shape)  # alpha
    aa = np.concatenate((aa, aa[:1, :]))
    bb = aa.copy()  # beta
    with np.errstate(invalid="ignore"):
        lcond = esarr[0, :] > anarr
        aa[0, lcond] = 1.0
        bb[0, :] = np.where(lcond, esarr[0, :] - anarr, 0.0)
        aa[1:-1, :] = np.where(esarr[1:, :] <= anarr, esarr[1:, :] - esarr[:-1, :], anarr - esarr[:-1, :])
        aa[1:-1, :][esarr[: n_ens - 1, :] > anarr] = 0.0  # this would be hard in xarray
        bb[1:-1, :] = np.where(esarr[:-1, :] > anarr, esarr[1:, :] - esarr[:-1, :], esarr[1:, :] - anarr)
        bb[1:-1, :][esarr[1:, :] <= anarr] = 0.0
        lcond = anarr > esarr[-1, :]
        aa[-1, :] = np.where(lcond, anarr - esarr[-1, :], 0.0)
        bb[-1, lcond] = 1.0
    # back to xarrays
    # alpha = xarray.DataArray(aa, dims=e.dims)
    # beta = xarray.DataArray(bb, dims=e.dims)
    # weight = xarray.DataArray(np.arange(n_ens + 1), dims=ENS_DIM) / float(n_ens)
    # w = np.arange(n_ens+1)/float(n_ens)
    w = (np.arange(n_ens + 1) / float(n_ens)).reshape(n_ens + 1, *([1] * y.ndim))
    # w = np.expand_dims(np.arange(n_ens+1)/float(n_ens), axis=1)
    crps = aa * w**2 + bb * (1.0 - w) ** 2
    crps_sum = np.nansum(crps, axis=0)
    return crps_sum
    # Fair CRPS
    # fcrps = crps - self.ginis_mean_diff() / (2. * n_ens)
    # return alpha, beta, crps, fcrps


def _get_crps_data():
    here = os.path.dirname(__file__)
    sys.path.insert(0, here)
    from _crps import ens
    from _crps import obs
    from _crps import v_ref

    return obs, ens, v_ref


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("obs,ens,v_ref", [_get_crps_data()])
def test_crps_meteo(obs, ens, v_ref, array_backend):
    obs, ens, v_ref = array_backend.asarray(obs, ens, v_ref)

    c = score.crps(ens.T, obs[0])

    for i in range(ens.shape[0]):
        assert array_backend.isclose(c[i], v_ref[i]), f"i={i}"

    assert array_backend.isclose(array_backend.xp.mean(c), array_backend.xp.mean(v_ref))


@pytest.mark.parametrize("array_backend", get_array_backend(["numpy"]))
@pytest.mark.parametrize("obs,ens,v_ref", [_get_crps_data()])
def test_crps_quaver2(obs, ens, v_ref, array_backend):
    obs, ens, v_ref = array_backend.asarray(obs, ens, v_ref)

    c = crps_quaver2(ens.T, obs[0])

    for i in range(ens.shape[0]):
        assert array_backend.isclose(c[i], v_ref[i]), f"i={i}"

    assert array_backend.isclose(array_backend.xp.mean(c), array_backend.xp.mean(v_ref))
