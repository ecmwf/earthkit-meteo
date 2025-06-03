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
from earthkit.utils.testing import skip_array_backend

from earthkit.meteo import stats
from earthkit.meteo.utils.testing import ARRAY_BACKENDS

np.set_printoptions(formatter={"float_kind": "{:.10f}".format})


def _get_quantile_data():
    here = os.path.dirname(__file__)
    sys.path.insert(0, here)
    from _quantile import q_test_array

    return q_test_array


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
# TODO: this! test fails with cupy. The reason is that cupy.quantile works differently
#       than np.quantile when nans are present
@pytest.mark.parametrize("array_backend", skip_array_backend(ARRAY_BACKENDS, "cupy"))
@pytest.mark.parametrize("arr", [_get_quantile_data()])
def test_quantiles_nans(arr, array_backend):
    arr = array_backend.asarray(arr)
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    r1 = [quantile for quantile in stats.iter_quantiles(arr, qs, method="sort")]
    r2 = [quantile for quantile in stats.iter_quantiles(arr, qs, method="numpy")]
    for i, (d1, d2) in enumerate(zip(r1, r2)):
        assert array_backend.allclose(d1, d2, equal_nan=True), f"quantile={qs[i]}"


def test_GumbelDistribution():
    dist = stats.GumbelDistribution.fit([6.0, 5.0, 6.0, 7.0, 9.0, 5.0, 6.0, 7.0])
    # Extreme ends of distribution
    np.testing.assert_allclose(dist.cdf(0.0), 1.0)
    np.testing.assert_allclose(dist.cdf(100.0), 0.0)
    # Test identity when calling method followed by inverse method
    values = np.linspace(4.0, 10.0, 21)
    np.testing.assert_allclose(dist.ppf(dist.cdf(values)), values)
    probs = np.linspace(0.01, 0.99, 21)
    np.testing.assert_allclose(dist.cdf(dist.ppf(probs)), probs)


def test_GumbelDistribution_along_axis():
    sample = [
        [0.3, 3.0, 30.0],
        [0.5, 5.0, 50.0],
        [0.7, 7.0, 70.0],
        [0.3, 3.0, 30.0],
        [0.4, 4.0, 40.0],
        [0.6, 6.0, 60.0],
        [0.7, 7.0, 70.0],
    ]
    dist = stats.GumbelDistribution.fit(sample, axis=0)
    values = dist.ppf([0.01, 0.3, 0.5, 0.6, 0.99])
    assert values.shape == (5, 3)
    # Results should scale with values
    np.testing.assert_allclose(values[:, 0] * 10.0, values[:, 1])
    np.testing.assert_allclose(values[:, 1] * 10.0, values[:, 2])


def test_return_period_identity():
    dist = stats.GumbelDistribution.fit([6.0, 5.0, 6.0, 7.0, 9.0, 5.0, 6.0, 7.0])
    values = np.linspace(4.0, 10.0, 21)
    np.testing.assert_allclose(
        stats.return_period_to_value(dist, stats.value_to_return_period(dist, values)), values
    )


def test_return_period_with_frequency():
    freq = np.timedelta64(24 * 60 * 60, "s")
    sample = [4.0, 3.0, 5.5, 6.0, 7.0, 5.3, 2.1]
    dist = stats.GumbelDistribution.fit(sample, freq=freq)
    # Return periods should be scaled by the given frequency
    assert stats.value_to_return_period(dist, 6.0).dtype == freq.dtype
