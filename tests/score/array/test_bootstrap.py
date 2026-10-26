# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import TypeVar

import numpy as np
import pytest

from earthkit.meteo.score import array as bootstrap


def _create_rng(
    num: int, size: int, seed: int, gen: type[np.random.BitGenerator] | None = None, replace: bool = True
) -> tuple[np.random.Generator, np.ndarray]:
    make_rng = np.random.default_rng if gen is None else (lambda s: np.random.Generator(gen(s)))
    rng = make_rng(seed)
    seq = rng.choice(num, size=size, replace=replace)
    return make_rng(seed), seq


T = TypeVar("T")
U = TypeVar("U")


def _populate_kwarg(name: str, val: T | None, default: U, kwargs: dict) -> T | U:
    if val is None:
        val = default
    else:
        kwargs[name] = val
    return val


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, None), (6, True), (6, False), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_iter_samples(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 10

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(n_inputs)
    samples = list(bootstrap.iter_samples(x, rng=rng, **kwargs))
    assert len(samples) == n_iter
    assert len(samples[0]) == 1
    assert len(samples[0][0]) == n_samples
    np.testing.assert_equal(samples[0][0], seq)


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, None), (6, True), (6, False), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_iter_samples_2(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 10

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(n_inputs)
    y = 10 - x
    samples = list(bootstrap.iter_samples(x, y, rng=rng, **kwargs))
    assert len(samples) == n_iter
    assert len(samples[0]) == 2
    assert len(samples[0][0]) == n_samples
    assert len(samples[0][1]) == n_samples
    np.testing.assert_equal(samples[0][0], seq)
    for sx, sy in samples:
        np.testing.assert_equal(sy, 10 - sx)


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_iter_samples_dim(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(20).reshape(4, n_inputs)
    samples = list(bootstrap.iter_samples(x, dim=1, rng=rng, **kwargs))
    assert len(samples) == n_iter
    assert len(samples[0]) == 1
    assert samples[0][0].shape == (4, n_samples)
    np.testing.assert_equal(
        samples[0][0],
        [seq, seq + 5, seq + 10, seq + 15],
    )


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_iter_samples_dim_2(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(20).reshape(4, n_inputs)
    y = 20 - x
    samples = list(bootstrap.iter_samples(x, y, dim=1, rng=rng, **kwargs))
    assert len(samples) == n_iter
    assert len(samples[0]) == 2
    assert samples[0][0].shape == (4, n_samples)
    assert samples[0][1].shape == (4, n_samples)
    np.testing.assert_equal(
        samples[0][0],
        [seq, seq + 5, seq + 10, seq + 15],
    )
    for sx, sy in samples:
        np.testing.assert_equal(sy, 20 - sx)


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_iter_samples_dim_2diff(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(20).reshape(4, n_inputs)
    y = np.arange(n_inputs)
    samples = list(bootstrap.iter_samples(x, y, dim=[1, 0], rng=rng, **kwargs))
    assert len(samples) == n_iter
    assert len(samples[0]) == 2
    assert samples[0][0].shape == (4, n_samples)
    assert samples[0][1].shape == (n_samples,)
    np.testing.assert_equal(
        samples[0][0],
        [seq, seq + 5, seq + 10, seq + 15],
    )
    for sx, sy in samples:
        np.testing.assert_equal(sy, sx[0])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, None), (6, True), (6, False), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_resample(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 10

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(n_inputs)
    samples = bootstrap.resample(x, rng=rng, **kwargs)
    assert len(samples) == 1
    assert samples[0].shape == (n_iter, n_samples)
    np.testing.assert_equal(samples[0][0], seq)


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, None), (6, True), (6, False), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_resample_2(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 10

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(n_inputs)
    y = 10 - x
    samples = bootstrap.resample(x, y, rng=rng, **kwargs)
    assert len(samples) == 2
    assert samples[0].shape == (n_iter, n_samples)
    assert samples[1].shape == (n_iter, n_samples)
    np.testing.assert_equal(samples[0][0], seq)
    np.testing.assert_equal(samples[1], 10 - samples[0])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_resample_dim(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(20).reshape(4, n_inputs)
    samples = bootstrap.resample(x, dim=1, rng=rng, **kwargs)
    assert len(samples) == 1
    assert samples[0].shape == (n_iter, 4, n_samples)
    np.testing.assert_equal(
        samples[0][0],
        [seq, seq + 5, seq + 10, seq + 15],
    )


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_resample_dim_2(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(20).reshape(4, n_inputs)
    y = 20 - x
    samples = bootstrap.resample(x, y, dim=1, rng=rng, **kwargs)
    assert len(samples) == 2
    assert samples[0].shape == (n_iter, 4, n_samples)
    assert samples[1].shape == (n_iter, 4, n_samples)
    np.testing.assert_equal(
        samples[0][0],
        [seq, seq + 5, seq + 10, seq + 15],
    )
    np.testing.assert_equal(samples[1], 20 - samples[0])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_resample_dim_2diff(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(20).reshape(4, n_inputs)
    y = np.arange(n_inputs)
    samples = bootstrap.resample(x, y, dim=[1, 0], rng=rng, **kwargs)
    assert len(samples) == 2
    assert samples[0].shape == (n_iter, 4, n_samples)
    assert samples[1].shape == (n_iter, n_samples)
    np.testing.assert_equal(
        samples[0][0],
        [seq, seq + 5, seq + 10, seq + 15],
    )
    np.testing.assert_equal(samples[1], samples[0][:, 0, :])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, None), (6, True), (6, False), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_resample_out_dim(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 10

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(n_inputs)
    samples = bootstrap.resample(x, out_dim=1, rng=rng, **kwargs)
    assert len(samples) == 1
    assert samples[0].shape == (n_samples, n_iter)
    np.testing.assert_equal(samples[0][:, 0], seq)


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_resample_dim_out_dim(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(20).reshape(4, n_inputs)
    samples = bootstrap.resample(x, dim=1, out_dim=1, rng=rng, **kwargs)
    assert len(samples) == 1
    assert samples[0].shape == (4, n_iter, n_samples)
    np.testing.assert_equal(samples[0][0, 0, :], seq)


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, None), (6, True), (6, False), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_bootstrap(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 10

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(n_inputs)
    values = bootstrap.bootstrap(np.mean, x, rng=rng, **kwargs)
    assert values.shape == (n_iter,)
    np.testing.assert_allclose(values[0], np.mean(seq))


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, None), (6, True), (6, False), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_bootstrap_2(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 10

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(n_inputs)
    y = 10 - x
    values = bootstrap.bootstrap((lambda a, b: np.mean(a) - np.mean(b)), x, y, rng=rng, **kwargs)
    assert values.shape == (n_iter,)
    np.testing.assert_allclose(values[0], 2 * np.mean(seq) - 10)


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_bootstrap_dim(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(20).reshape(4, n_inputs)
    values = bootstrap.bootstrap((lambda a: np.mean(a, axis=1)), x, dim=1, rng=rng, **kwargs)
    assert values.shape == (n_iter, 4)
    m = np.mean(seq)
    np.testing.assert_allclose(values[0], [m, m + 5, m + 10, m + 15])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_bootstrap_dim_2(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(20).reshape(4, n_inputs)
    y = 20 - x
    values = bootstrap.bootstrap((lambda a, b: np.mean(a, axis=1) - np.mean(b, axis=1)), x, y, dim=1, rng=rng, **kwargs)
    assert values.shape == (n_iter, 4)
    m = np.mean(seq)
    np.testing.assert_allclose(values[0], [2 * m - 20, 2 * m - 10, 2 * m, 2 * m + 10])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_bootstrap_dim_2diff(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, _ = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(20).reshape(4, n_inputs)
    y = np.arange(n_inputs)
    values = bootstrap.bootstrap((lambda a, b: np.mean(a, axis=1) - np.mean(b)), x, y, dim=[1, 0], rng=rng, **kwargs)
    assert values.shape == (n_iter, 4)
    np.testing.assert_allclose(values[0], [0, 5, 10, 15])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_bootstrap_dim_out_dim(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = np.arange(20).reshape(4, n_inputs)
    values = bootstrap.bootstrap((lambda a: np.mean(a, axis=1)), x, dim=1, out_dim=1, rng=rng, **kwargs)
    assert values.shape == (4, n_iter)
    m = np.mean(seq)
    np.testing.assert_allclose(values[:, 0], [m, m + 5, m + 10, m + 15])
