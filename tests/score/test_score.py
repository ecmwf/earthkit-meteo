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
@pytest.mark.parametrize("nan_policy", ["raise", "propagate", "omit"])
@pytest.mark.parametrize("obs,ens,v_ref", [_get_crps_data()])
def test_crps_meteo(obs, ens, v_ref, array_backend, nan_policy):
    obs, ens, v_ref = array_backend.asarray(obs, ens, v_ref)
    xp = array_backend.namespace

    c = score.crps(ens.T, obs[0], nan_policy)

    for i in range(ens.shape[0]):
        assert array_backend.isclose(c[i], v_ref[i]), f"i={i}"

    assert array_backend.isclose(xp.mean(c), xp.mean(v_ref))


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("nan_policy", ["raise", "propagate", "omit"])
@pytest.mark.parametrize("obs,ens,v_ref", [_get_crps_data()])
def test_crps_meteo_missing(obs, ens, v_ref, array_backend, nan_policy):
    obs, ens, v_ref = array_backend.asarray(obs, ens, v_ref)
    xp = array_backend.namespace

    ens = ens.T
    obs = obs[0]
    obs[::2] = xp.nan
    ens[0, ::3] = xp.nan

    nan_mask = xp.any(xp.isnan(ens), axis=0) | xp.isnan(obs)

    if nan_policy == "raise":
        with pytest.raises(ValueError):
            score.crps(ens, obs, nan_policy)
    else:
        c_all = score.crps(ens, obs, nan_policy)
        c_non_missing = score.crps(ens[..., ~nan_mask], obs[~nan_mask])

        if nan_policy == "omit":
            for i in range(c_all.shape[0]):
                assert array_backend.isclose(c_all[i], c_non_missing[i])
        elif nan_policy == "propagate":
            j = 0
            for i in range(c_all.shape[0]):
                if nan_mask[i]:
                    assert xp.isnan(c_all[i])
                else:
                    assert array_backend.isclose(c_all[i], c_non_missing[j])
                    j += 1

        non_missing_crps = c_all[~xp.isnan(c_all)]
        assert array_backend.isclose(xp.mean(non_missing_crps), xp.mean(c_non_missing))


@pytest.mark.parametrize("array_backend", get_array_backend(["numpy"]))
@pytest.mark.parametrize("obs,ens,v_ref", [_get_crps_data()])
def test_crps_quaver2(obs, ens, v_ref, array_backend):
    obs, ens, v_ref = array_backend.asarray(obs, ens, v_ref)
    xp = array_backend.namespace

    c = crps_quaver2(ens.T, obs[0])

    for i in range(ens.shape[0]):
        assert array_backend.isclose(c[i], v_ref[i]), f"i={i}"

    assert array_backend.isclose(xp.mean(c), xp.mean(v_ref))


def _get_pearson_data():
    here = os.path.dirname(__file__)
    sys.path.insert(0, here)
    from _pearson import SAMPLE_X
    from _pearson import SAMPLE_Y

    rs = np.array([1.0, -1.0, 0.0, 0.42, -0.13, np.nan])

    SAMPLE_X = np.array(SAMPLE_X)
    SAMPLE_Y = np.array(SAMPLE_Y)

    crs = np.sqrt(1 - rs**2)
    ymiss = SAMPLE_Y.copy()
    ymiss[12:20] = np.nan
    x = np.vstack([SAMPLE_X, SAMPLE_X, SAMPLE_X, SAMPLE_Y, SAMPLE_X, SAMPLE_X])
    y = np.vstack(
        [
            SAMPLE_X,
            -SAMPLE_X,
            SAMPLE_Y,
            crs[3] * SAMPLE_X + rs[3] * SAMPLE_Y,
            rs[4] * SAMPLE_X + crs[4] * SAMPLE_Y,
            ymiss,
        ]
    )
    return x.tolist(), y.tolist(), rs.tolist()


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("x, y, v_ref", [_get_pearson_data()])
def test_pearson(x, y, v_ref, array_backend):
    x, y, v_ref = array_backend.asarray(x, y, v_ref)

    r = score.pearson(x, y, axis=1)
    np.testing.assert_allclose(r, v_ref, atol=1e-7)
