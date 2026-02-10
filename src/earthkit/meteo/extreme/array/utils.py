# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from __future__ import annotations

from typing import Any


def flatten_extreme_input(xp: Any, da: Any, axis: int) -> tuple[Any, tuple[int, ...]]:
    """Move the reduction axis to position 0 and flatten the remaining dimensions."""
    da = xp.moveaxis(da, axis, 0)
    out_shape = da.shape[1:]

    npoints = 1
    for s in out_shape:
        npoints *= s

    da = xp.reshape(da, (da.shape[0], npoints))
    return da, out_shape


def validate_extreme_shapes(
    *,
    func: str,
    clim_shape: tuple[int, ...],
    ens_shape: tuple[int, ...],
    clim_axis: int,
    ens_axis: int,
) -> None:
    """Validate that clim and ens shapes match in shape"""
    clim_shape_tmp = list(clim_shape)
    clim_shape_tmp.pop(clim_axis)
    ens_shape_tmp = list(ens_shape)
    ens_shape_tmp.pop(ens_axis)
    if clim_shape_tmp != ens_shape_tmp:
        raise ValueError(
            f"{func}(): clim and ens must match in shape "
            f"{clim_axis=} {ens_axis=}. {clim_shape=}, {ens_shape=}"
        )
