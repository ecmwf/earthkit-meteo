# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from collections.abc import Callable

import numpy.random as npr
import xarray as xr


def resample(
    *args: xr.DataArray,
    dim: str = None,
    out_dim: str = "sample",
    n_iter: int = 100,
    n_samples: int | None = None,
    replace: bool = True,
    rng: npr.Generator | None = None,
) -> tuple[xr.DataArray, ...]:
    """Resample arrays for bootstrapping.

    Parameters
    ----------
    *args: xarray object
        Arrays to sample. Must have the same size along ``dim``
    dim: str
        Sample along this dimension
    out_dim: str
        Output dimension name for samples
    n_iter: int
        Number of bootstrapping iterations
    n_samples: int or None
        Number of samples for each iteration. If None, use the number of
        inputs (size of the first array along the sampling dimension)
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
        raise TypeError("resample with xarray arguments requires 'dim'")
    if out_dim is None:
        out_dim = "sample"
    n_inputs = args[0].sizes[dim]
    assert all(arr.sizes[dim] == n_inputs for arr in args), (
        "Input arrays must have the same size along the sampling axis"
    )
    if n_samples is None:
        n_samples = n_inputs
    if rng is None:
        rng = npr.default_rng()
    n_arrays = len(args)
    samples = [[] for _ in range(n_arrays)]
    for _ in range(n_iter):
        indices = rng.choice(n_inputs, size=n_samples, replace=replace)
        for i in range(n_arrays):
            samples[i].append(args[i].isel({dim: indices}))
    return tuple(xr.concat(sampled_arr, out_dim) for sampled_arr in samples)


def bootstrap(
    func: Callable[..., xr.DataArray],
    *args: xr.DataArray,
    dim: str = None,
    out_dim: str = "sample",
    n_iter: int = 100,
    n_samples: int | None = None,
    replace: bool = True,
    rng: npr.Generator | None = None,
    **kwargs,
) -> xr.DataArray:
    """Run bootstrapping.

    Parameters
    ----------
    func: function ((array, ..., **kwargs) -> array)
        Function to bootstrap
    *args: xarray object
        Inputs to ``function``, sampled for bootstrapping. Must have the same
        size along ``dim``
    dim: str
        Sample along this dimension
    out_dim: str
        Output dimension name for samples
    n_iter: int
        Number of bootstrapping iterations
    n_samples: int or None
        Number of samples for each iteration. If None, use the number of
        inputs (size of the first array along the sampling dimension)
    replace: bool
        Sample with replacement (on by default)
    rng: numpy.random.Generator
        Random number generator
    **kwargs
        Additional keyword arguments to ``func``

    Returns
    -------
    xarray object
        Aggregated results of the bootstrapping process
    """
    if dim is None:
        raise TypeError("bootstrap with xarray arguments requires 'dim'")
    if out_dim is None:
        out_dim = "sample"
    n_inputs = args[0].sizes[dim]
    assert all(arr.sizes[dim] == n_inputs for arr in args), (
        "Input arrays must have the same size along the sampling axis"
    )
    if n_samples is None:
        n_samples = n_inputs
    if rng is None:
        rng = npr.default_rng()
    results = []
    for _ in range(n_iter):
        indices = rng.choice(n_inputs, size=n_samples, replace=replace)
        sampled = tuple(arr.isel({dim: indices}) for arr in args)
        results.append(func(*sampled, **kwargs))
    return xr.concat(results, out_dim)
