from typing import Optional
from typing import TypeVar

import xarray as xr

T = TypeVar("T", xr.DataArray, xr.Dataset)

def nanaverage(data: T, weights: Optional[T] = None, **kwargs):
    if weights is not None:
        return data.weighted(weights).mean(**kwargs)
    else:
        return data.mean(**kwargs)