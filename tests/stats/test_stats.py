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

from earthkit.meteo import stats
from earthkit.meteo.utils.testing import ARRAY_BACKENDS

# from earthkit.meteo.utils.testing import get_array_backend


np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize(
    "data,weights,kwargs,v_ref",
    [
        (
            [[[4, 2], [4, np.nan]], [[4, 4], [np.nan, 2]]],
            None,
            dict(axis=-1),
            [[3.0, 4.0], [4.0, 2.0]],
        ),
        (
            [[[4, 2], [4, np.nan]], [[4, 4], [np.nan, 2]]],
            [1, 0.25],
            dict(axis=-1),
            [[3.6, 4.0], [4.0, 2.0]],
        ),
    ],
)
def test_nanaverage(data, weights, v_ref, kwargs, array_backend):
    data, v_ref = array_backend.asarray(data, v_ref)

    if weights is not None:
        weights = array_backend.asarray(weights)

    r = stats.nanaverage(data, weights=weights, **kwargs)
    assert array_backend.allclose(r, v_ref)

    # NOTE: we used the following numpy code to compute the reference values!
    # when weight is None:
    #    v_ref = np.nanmean(data, axis=-1)
    # when weight is not None:
    #   v_ref = np.average(data, axis=-1, weights=weights)
    #   v_ref[:, 1] = np.nansum(data, axis=-1)[:, 1]

    # v_ref = np.nanmean(data, axis=-1)

    # v_ref = np.average(data, axis=-1, weights=weights)
    # # replace nan values
    # v_ref[:, 1] = np.nansum(data, axis=-1)[:, 1]


@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
@pytest.mark.parametrize("method", ["sort", "numpy_bulk", "numpy"])
@pytest.mark.parametrize(
    "data,which,kwargs,v_ref",
    [
        (
            [
                [3, 6, 12, 0, 45, 0, 4, 0, 7],
                [1, 2.5, 4, 3, 18, 3, 8, 7, 2],
                [3, 4, 9, 0.5, 2, 48, 4, 9, 6],
                [5, 19, 8.3, 2, 6, 1, 0, 1, 0],
            ],
            4,
            dict(axis=1),
            [
                [0, 1, 0.5, 0],
                [0, 2.5, 3, 1],
                [4, 3, 4, 2],
                [7, 7, 9, 6],
                [45, 18, 48, 19],
            ],
        ),
        (
            [
                [3, 6, 12, 0, 45, 0, 4, 0, 7],
                [1, 2.5, 4, 3, 18, 3, 8, 7, 2],
                [3, 4, 9, 0.5, 2, 48, 4, 9, 6],
                [5, 19, 8.3, 2, 6, 1, 0, 1, 0],
            ],
            [0.5, 1.0],
            {},
            # TODO: check if the commented or uncommented data block below is the
            # correct reference data. The commented data block is the one that
            # was present in the original test, but the results were not tested
            # against it, so it is not clear if it is correct.
            # [
            #     [3, 5, 6.15, 1.25, 12, 2, 4, 4, 4],
            #     [5, 19, 12, 3, 45, 48, 8, 9, 7],
            # ],
            [
                [
                    3.0000000000,
                    5.0000000000,
                    8.6500000000,
                    1.2500000000,
                    12.0000000000,
                    2.0000000000,
                    4.0000000000,
                    4.0000000000,
                    4.0000000000,
                ],
                [
                    5.0000000000,
                    19.0000000000,
                    12.0000000000,
                    3.0000000000,
                    45.0000000000,
                    48.0000000000,
                    8.0000000000,
                    9.0000000000,
                    7.0000000000,
                ],
            ],
        ),
    ],
)
def test_quantiles_core(data, which, kwargs, v_ref, method, array_backend):
    data, v_ref = array_backend.asarray(data, v_ref)

    r = list(stats.iter_quantiles(data, which, method=method, **kwargs))
    assert len(r) == v_ref.shape[0]
    for i, d in enumerate(r):
        # this is needed to handle the case where the last dimension is 1
        if d.ndim >= 2 and d.shape[-1] == 1:
            d = d[..., 0]

        assert array_backend.allclose(d, v_ref[i], rtol=1e-4), f"i={i}, d={d}, v_ref={v_ref[i]}"


# TODO: reimplement this test to use reference values
@pytest.mark.parametrize("array_backend", ARRAY_BACKENDS)
def test_quantiles_nans(array_backend):
    arr = np.random.rand(100, 100, 100)
    arr.ravel()[np.random.choice(arr.size, 100000, replace=False)] = np.nan

    arr = array_backend.asarray(arr)
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]

    r1 = [quantile for quantile in stats.iter_quantiles(arr, qs, method="sort")]
    r2 = [quantile for quantile in stats.iter_quantiles(arr, qs, method="numpy")]
    for d1, d2 in zip(r1, r2):
        assert array_backend.allclose(d1, d2, equal_nan=True)

    # assert np.all(np.isclose(sort, numpy, equal_nan=True))

    # arr = np.random.rand(100, 100, 100)
    # arr.ravel()[np.random.choice(arr.size, 100000, replace=False)] = np.nan
    # qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    # sort = [quantile for quantile in stats.iter_quantiles(arr.copy(), qs, method="sort")]
    # numpy = [quantile for quantile in stats.iter_quantiles(arr.copy(), qs, method="numpy")]
    # assert np.all(np.isclose(sort, numpy, equal_nan=True))
