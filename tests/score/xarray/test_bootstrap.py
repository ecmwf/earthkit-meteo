from __future__ import annotations

from typing import TypeVar

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from earthkit.meteo.score import xarray as bootstrap


def _create_rng(
    num: int, size: int, seed: int, gen: type[np.random.BitGenerator] | None = None, replace: bool = True
) -> tuple[np.random.Generator, np.ndarray]:
    make_rng = np.random.default_rng if gen is None else (lambda s: np.random.Generator(gen(s)))
    rng = make_rng(seed)
    seq = rng.choice(num, size=size, replace=replace)
    return make_rng(seed), xr.DataArray(seq, dims=["number"])


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
def test_resample(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 10

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = xr.DataArray(np.arange(n_inputs), dims=["number"])
    samples = bootstrap.resample(x, dim="number", rng=rng, **kwargs)
    assert len(samples) == 1
    assert samples[0].dims == ("sample", "number")
    assert samples[0].sizes["number"] == n_samples
    assert samples[0].sizes["sample"] == n_iter
    xr.testing.assert_allclose(samples[0].sel(sample=0), seq)


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

    x = xr.DataArray(np.arange(n_inputs), dims=["number"])
    y = 10 - x
    samples = bootstrap.resample(x, y, dim="number", rng=rng, **kwargs)
    assert len(samples) == 2
    assert samples[0].dims == ("sample", "number")
    assert samples[0].sizes["sample"] == n_iter
    assert samples[0].sizes["number"] == n_samples
    assert samples[1].dims == ("sample", "number")
    assert samples[1].sizes["sample"] == n_iter
    assert samples[1].sizes["number"] == n_samples
    xr.testing.assert_allclose(samples[0].sel(sample=0), seq)
    xr.testing.assert_allclose(samples[1], 10 - samples[0])


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_resample_2d(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = xr.DataArray(np.arange(20).reshape(4, n_inputs), dims=["point", "number"])
    samples = bootstrap.resample(x, dim="number", rng=rng, **kwargs)
    assert len(samples) == 1
    assert samples[0].dims == ("sample", "point", "number")
    assert samples[0].sizes["sample"] == n_iter
    assert samples[0].sizes["point"] == 4
    assert samples[0].sizes["number"] == n_samples
    xr.testing.assert_allclose(
        samples[0].sel(sample=0),
        xr.concat([seq, seq + 5, seq + 10, seq + 15], "point"),
    )


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_resample_2diff(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = xr.DataArray(np.arange(20).reshape(4, n_inputs), dims=["point", "number"])
    y = xr.DataArray(np.arange(n_inputs), dims=["number"])
    samples = bootstrap.resample(x, y, dim="number", rng=rng, **kwargs)
    assert len(samples) == 2
    assert samples[0].dims == ("sample", "point", "number")
    assert samples[0].sizes["sample"] == n_iter
    assert samples[0].sizes["point"] == 4
    assert samples[0].sizes["number"] == n_samples
    assert samples[1].dims == ("sample", "number")
    assert samples[1].sizes["sample"] == n_iter
    assert samples[1].sizes["number"] == n_samples
    seq = seq.isel(number=slice(None, n_samples))
    xr.testing.assert_allclose(
        samples[0].sel(sample=0),
        xr.concat([seq, seq + 5, seq + 10, seq + 15], "point"),
    )
    xr.testing.assert_allclose(samples[1], samples[0].sel(point=0))


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

    x = xr.DataArray(np.arange(n_inputs), dims=["number"])
    samples = bootstrap.resample(x, dim="number", out_dim="iteration", rng=rng, **kwargs)
    assert len(samples) == 1
    assert samples[0].dims == ("iteration", "number")
    assert samples[0].sizes["number"] == n_samples
    assert samples[0].sizes["iteration"] == n_iter
    xr.testing.assert_allclose(samples[0].sel(iteration=0), seq)


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

    x = xr.DataArray(np.arange(n_inputs), dims=["number"])
    values = bootstrap.bootstrap((lambda a: a.mean("number")), x, dim="number", rng=rng, **kwargs)
    assert values.dims == ("sample",)
    assert values.sizes["sample"] == n_iter
    xr.testing.assert_allclose(values.sel(sample=0), seq.mean("number"))


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

    x = xr.DataArray(np.arange(n_inputs), dims=["number"])
    y = 10 - x
    values = bootstrap.bootstrap(
        (lambda a, b: a.mean("number") - b.mean("number")), x, y, dim="number", rng=rng, **kwargs
    )
    assert values.dims == ("sample",)
    assert values.sizes["sample"] == n_iter
    xr.testing.assert_allclose(values.sel(sample=0), 2 * seq.mean("number") - 10)


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_bootstrap_2d(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = xr.DataArray(np.arange(20).reshape(4, n_inputs), dims=["point", "number"])
    values = bootstrap.bootstrap((lambda a: a.mean("number")), x, dim="number", rng=rng, **kwargs)
    assert values.dims == ("sample", "point")
    assert values.sizes["sample"] == n_iter
    assert values.sizes["point"] == 4
    m = seq.mean("number")
    xr.testing.assert_allclose(values.sel(sample=0), xr.concat([m, m + 5, m + 10, m + 15], "point"))


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, True), (2, False), (6, True), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_bootstrap_2diff(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 5

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, _ = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = xr.DataArray(np.arange(20).reshape(4, n_inputs), dims=["point", "number"])
    y = xr.DataArray(np.arange(n_inputs), dims=["number"])
    values = bootstrap.bootstrap(
        (lambda a, b: a.mean("number") - b.mean("number")), x, y, dim="number", rng=rng, **kwargs
    )
    assert values.dims == ("sample", "point")
    assert values.sizes["sample"] == n_iter
    assert values.sizes["point"] == 4
    xr.testing.assert_allclose(values.sel(sample=0), xr.DataArray([0, 5, 10, 15], dims=["point"]))


@pytest.mark.parametrize("n_iter", [None, 1, 5])
@pytest.mark.parametrize("n_samples, replace", [(None, None), (2, None), (6, True), (6, False), (15, True)])
@pytest.mark.parametrize("gen", [None, np.random.MT19937], ids=["default", "custom"])
def test_bootstrap_out_dim(
    n_iter: int | None, n_samples: int | None, replace: bool | None, gen: type[np.random.BitGenerator] | None
):
    n_inputs = 10

    kwargs = {}
    n_iter = _populate_kwarg("n_iter", n_iter, 100, kwargs)
    n_samples = _populate_kwarg("n_samples", n_samples, n_inputs, kwargs)
    replace = _populate_kwarg("replace", replace, True, kwargs)
    rng, seq = _create_rng(n_inputs, n_samples, 2, gen, replace=replace)

    x = xr.DataArray(np.arange(n_inputs), dims=["number"])
    values = bootstrap.bootstrap((lambda a: a.mean("number")), x, dim="number", out_dim="iteration", rng=rng, **kwargs)
    assert values.dims == ("iteration",)
    assert values.sizes["iteration"] == n_iter
    xr.testing.assert_allclose(values.sel(iteration=0), seq.mean("number"))
