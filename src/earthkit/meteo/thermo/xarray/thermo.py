# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import Iterable

import numpy as np

from .. import array


def _sample_arg(arg: object) -> np.ndarray:
    if hasattr(arg, "dtype"):
        return np.zeros((), dtype=arg.dtype)
    return np.zeros((), dtype=float)


def _infer_output_dtypes(func, *args, **kwargs) -> list[np.dtype]:
    sample_args = [_sample_arg(arg) for arg in args]
    res = func(*sample_args, **kwargs)
    if isinstance(res, tuple):
        return [np.asarray(item).dtype for item in res]
    return [np.asarray(res).dtype]


def _apply_ufunc(func, *args, **kwargs):
    import xarray as xr

    output_dtypes = _infer_output_dtypes(func, *args, **kwargs)

    return xr.apply_ufunc(
        func,
        *args,
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=output_dtypes,
        keep_attrs=True,
    )
