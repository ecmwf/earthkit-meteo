# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias, overload

from ..utils.decorators import dispatch

if TYPE_CHECKING:
    import numpy.random as npr  # type: ignore[import]
    import xarray  # type: ignore[import]

ArrayLike: TypeAlias = Any


@overload
def resample(
    x: ArrayLike,
    *args: ArrayLike,
    dim: int | list[int] = 0,
    out_dim: int = 0,
    n_iter: int = 100,
    n_samples: int | None = None,
    replace: bool = True,
    rng: "npr.Generator | None" = None,
) -> tuple[ArrayLike, ...]: ...


@overload
def resample(
    x: "xarray.DataArray",
    *args: "xarray.DataArray",
    dim: str = None,
    out_dim: str = "sample",
    n_iter: int = 100,
    n_samples: int | None = None,
    replace: bool = True,
    rng: "npr.Generator | None" = None,
) -> tuple["xarray.DataArray", ...]: ...


def resample(
    x: "ArrayLike | xarray.DataArray",
    *args: "ArrayLike | xarray.DataArray",
    dim: int | list[int] | str = None,
    out_dim: int | str = None,
    n_iter: int = 100,
    n_samples: int | None = None,
    replace: bool = True,
    rng: "npr.Generator | None" = None,
    **kwargs,
) -> tuple[ArrayLike, ...] | tuple["xarray.DataArray", ...]:
    """Resample arrays for bootstrapping.

    Parameters
    ----------
    x, *args: xarray object or array-like
        Arrays to sample. Must have the same size along ``dim``
    dim: str or int or list of int
        Sample along this dimension (name or index/indices for array-like)
    out_dim: str or int
        Output dimension name (or index for array-like) for samples
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
    dispatched = dispatch(resample, fieldlist=False, array=True)
    return dispatched(
        x,
        *args,
        dim=dim,
        out_dim=out_dim,
        n_iter=n_iter,
        n_samples=n_samples,
        replace=replace,
        rng=rng,
        **kwargs,
    )


@overload
def bootstrap(
    func: Callable[..., ArrayLike],
    x: ArrayLike,
    *args: ArrayLike,
    dim: int | list[int] = 0,
    out_dim: int = 0,
    n_iter: int = 100,
    n_samples: int | None = None,
    replace: bool = True,
    rng: "npr.Generator | None" = None,
    **kwargs,
) -> ArrayLike: ...


@overload
def bootstrap(
    func: Callable[..., "xarray.DataArray"],
    *args: "xarray.DataArray",
    dim: str = None,
    out_dim: str = "sample",
    n_iter: int = 100,
    n_samples: int | None = None,
    replace: bool = True,
    rng: "npr.Generator | None" = None,
    **kwargs,
) -> "xarray.DataArray": ...


def bootstrap(
    func: Callable[..., ArrayLike] | Callable[..., "xarray.DataArray"],
    x: "ArrayLike | xarray.DataArray",
    *args: "ArrayLike | xarray.DataArray",
    dim: int | list[int] | str = None,
    out_dim: int | str = None,
    n_iter: int = 100,
    n_samples: int | None = None,
    replace: bool = True,
    rng: "npr.Generator | None" = None,
    **kwargs,
) -> "ArrayLike | xarray.DataArray":
    """Run bootstrapping.

    Parameters
    ----------
    func: function ((array, ..., **kwargs) -> array)
        Function to bootstrap
    x, *args: xarray object or array-like
        Inputs to ``function``, sampled for bootstrapping. Must have the same
        size along ``dim``
    dim: str or int or list of int
        Sample along this dimension (name or index/indices for array-like)
    out_dim: str or int
        Output dimension name (or index for array-like) for samples
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
    xarray object or array-like
        Aggregated results of the bootstrapping process
    """
    dispatched = dispatch(bootstrap, match=1, fieldlist=False, array=True)
    return dispatched(
        func,
        x,
        *args,
        dim=dim,
        out_dim=out_dim,
        n_iter=n_iter,
        n_samples=n_samples,
        replace=replace,
        rng=rng,
        **kwargs,
    )
