from __future__ import annotations

from typing import Any


def flatten_extreme_input(xp: Any, da: Any, dim: int) -> tuple[Any, tuple[int, ...]]:
    """Move the reduction dimension to position 0 and flatten the remaining dimensions."""
    da = xp.moveaxis(da, dim, 0)
    out_shape = da.shape[1:]
    da = xp.reshape(da, (da.shape[0], -1))
    return da, out_shape


def validate_extreme_shapes(
    *,
    func: str,
    clim_shape: tuple[int, ...],
    ens_shape: tuple[int, ...],
    clim_dim: int,
    ens_dim: int,
) -> None:
    """Validate that clim and ens shapes match in shape."""
    clim_shape_tmp = list(clim_shape)
    clim_shape_tmp.pop(clim_dim)
    ens_shape_tmp = list(ens_shape)
    ens_shape_tmp.pop(ens_dim)
    if clim_shape_tmp != ens_shape_tmp:
        raise ValueError(
            f"{func}(): clim and ens must match in shape {clim_dim=} {ens_dim=}. {clim_shape=}, {ens_shape=}"
        )
