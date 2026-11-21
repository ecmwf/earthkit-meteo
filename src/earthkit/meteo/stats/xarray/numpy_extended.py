# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, TypeVar

import xarray as xr

T = TypeVar("T", xr.DataArray, xr.Dataset)


def nanaverage(data: T, weights: Optional[T] = None, **kwargs):
    if weights is not None:
        return data.weighted(weights).mean(**kwargs)
    else:
        return data.mean(**kwargs)
