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

from earthkit.meteo import stats

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


def _get_quantile_data():
    here = os.path.dirname(__file__)
    sys.path.insert(0, here)
    from _quantile import q_test_array

    return q_test_array


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
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
            dict(dim=1),
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
def test_quantiles_core(xp, device, data, which, kwargs, v_ref, method):
    data, v_ref = xp.asarray(data, device=device), xp.asarray(v_ref, device=device)

    r = list(stats.iter_quantiles(data, which, method=method, **kwargs))
    assert len(r) == v_ref.shape[0]
    for i, d in enumerate(r):
        # this is needed to handle the case where the last dimension is 1
        if d.ndim >= 2 and d.shape[-1] == 1:
            d = d[..., 0]

        assert xp.allclose(d, v_ref[i], rtol=1e-4), f"i={i}, d={d}, v_ref={v_ref[i]}"


# TODO: reimplement this test to use reference values
# TODO: this! test fails with cupy. The reason is that cupy.quantile works differently
#       than np.quantile when nans are present
@pytest.mark.parametrize(
    "xp, device", list(filter(lambda x: x[0]._earthkit_array_namespace_name != "cupy", NAMESPACE_DEVICES))
)
@pytest.mark.parametrize("arr", [_get_quantile_data()])
def test_quantiles_nans(xp, device, arr):
    arr = xp.asarray(arr, device=device)
    qs = xp.asarray([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
    bulk = xp.asarray([quantile for quantile in stats.iter_quantiles(arr, qs, method="numpy_bulk")])
    r1 = xp.asarray([quantile for quantile in stats.iter_quantiles(arr, qs, method="sort")])
    assert bulk.shape == r1.shape
    assert xp.allclose(bulk, r1, equal_nan=True)
    r2 = xp.asarray([quantile for quantile in stats.iter_quantiles(arr, qs, method="numpy")])
    assert bulk.shape == r2.shape
    assert xp.allclose(bulk, r2, equal_nan=True)


@pytest.mark.parametrize("xp, device", NAMESPACE_DEVICES)
def test_quantiles_dim_argument(xp, device):
    arr = xp.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], device=device)
    got = xp.asarray([q for q in stats.iter_quantiles(arr, which=[0.5], dim=1, method="numpy")])[0]
    ref = xp.asarray([2.0, 4.0], device=device)
    assert xp.allclose(got, ref)
