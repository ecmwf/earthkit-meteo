# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from collections.abc import Callable, Generator
from typing import Any, TypeAlias

import numpy.random as npr
from earthkit.utils.array import array_namespace

ArrayLike: TypeAlias = Any


def iter_samples(
    x: ArrayLike,
    *args: ArrayLike,
    dim: int | list[int] = 0,
    n_iter: int = 100,
    n_samples: int | None = None,
    replace: bool = True,
    rng: npr.Generator | None = None,
) -> Generator[tuple[ArrayLike, ...], None, None]:
    """Iterate over resampled arrays for bootstrapping.

    Parameters
    ----------
    x, *args: array-like
        Arrays to sample. Must have the same size along ``dim``
    dim: int or list of int
        Sample along this dimension index (either same for all or one per argument)
    n_iter: int
        Number of bootstrapping iterations
    n_samples: int or None
        Number of samples for each iteration. If None, use the number of
        inputs (size of ``x`` along the sampling dimension)
    replace: bool
        Sample with replacement (on by default)
    rng: numpy.random.Generator
        Random number generator

    Yields
    ------
    tuple
        Resampled arrays (one element per input array, one yield per iteration)
    """
    args = (x,) + args
    n_arrays = len(args)
    if isinstance(dim, int):
        dim = [dim for _ in range(n_arrays)]
    else:
        assert len(dim) == n_arrays, "dim must have one element per input array"
    if rng is None:
        rng = npr.default_rng()
    xp = array_namespace(*args)
    device = xp.device(x)
    arrays = tuple((xp.asarray(arr, device=device), axis) for arr, axis in zip(args, dim))
    n_inputs = x.shape[dim[0]]
    assert all(y.shape[axis] == n_inputs for y, axis in arrays), (
        "Input arrays must have the same size along the sampling dimension"
    )
    if n_samples is None:
        n_samples = n_inputs
    for _ in range(n_iter):
        indices = xp.asarray(rng.choice(n_inputs, size=n_samples, replace=replace), device=device)
        sampled = tuple(xp.take(y, indices=indices, axis=axis) for y, axis in arrays)
        yield sampled


def resample(
    x: ArrayLike,
    *args: ArrayLike,
    dim: int | list[int] = 0,
    out_dim: int = 0,
    n_iter: int = 100,
    n_samples: int | None = None,
    replace: bool = True,
    rng: npr.Generator | None = None,
) -> tuple[ArrayLike, ...]:
    """Resample arrays for bootstrapping.

    Parameters
    ----------
    x, *args: array-like
        Arrays to sample. Must have the same size along ``dim``
    dim: int or list of int
        Sample along this dimension index (either same for all or one per argument)
    out_dim: int
        Stack samples along this dimension index
    n_iter: int
        Number of bootstrapping iterations
    n_samples: int or None
        Number of samples for each iteration. If None, use the number of
        inputs (size of ``x`` along the sampling dimension)
    replace: bool
        Sample with replacement (on by default)
    rng: numpy.random.Generator
        Random number generator

    Returns
    -------
    tuple
        Resampled arrays (one element per input array)
    """
    if dim is None:
        dim = 0
    if out_dim is None:
        out_dim = 0
    xp = array_namespace(x, *args)
    n_arrays = len(args) + 1
    samples = [[] for _ in range(n_arrays)]
    samples_it = iter_samples(x, *args, dim=dim, n_iter=n_iter, n_samples=n_samples, replace=replace, rng=rng)
    for sample in samples_it:
        for i in range(n_arrays):
            samples[i].append(sample[i])
    return tuple(xp.stack(sampled_arr, axis=out_dim) for sampled_arr in samples)


def bootstrap(
    func: Callable[..., ArrayLike],
    x: ArrayLike,
    *args: ArrayLike,
    dim: int | list[int] = 0,
    out_dim: int = 0,
    n_iter: int = 100,
    n_samples: int | None = None,
    replace: bool = True,
    rng: npr.Generator | None = None,
    **kwargs,
) -> ArrayLike:
    """Run bootstrapping.

    Parameters
    ----------
    func: function ((array, ..., **kwargs) -> array)
        Function to bootstrap
    x, *args: array-like
        Inputs to ``function``, sampled for bootstrapping. Must have the same
        size along ``dim``
    dim: int or list of int
        Sample along this dimension index (either same for all or one per argument)
    out_dim: int
        Stack samples along this dimension index
    n_iter: int
        Number of bootstrapping iterations
    n_samples: int or None
        Number of samples for each iteration. If None, use the number of
        inputs (size of ``x`` along the sampling dimension)
    replace: bool
        Sample with replacement (on by default)
    rng: numpy.random.Generator
        Random number generator
    **kwargs
        Additional keyword arguments to ``func``

    Returns
    -------
    array-like
        Aggregated results of the bootstrapping process
    """
    xp = array_namespace(x, *args)
    samples = iter_samples(
        x,
        *args,
        dim=dim,
        n_iter=n_iter,
        n_samples=n_samples,
        replace=replace,
        rng=rng,
    )
    return xp.stack([func(*sampled, **kwargs) for sampled in samples], axis=out_dim)
