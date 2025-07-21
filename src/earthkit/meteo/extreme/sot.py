# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import xarray as xr

from . import array  # noqa
from . import utils


def sot(
    clim: xr.DataArray | xr.Dataset,
    ens: xr.DataArray | xr.Dataset,
    perc: int,
    eps: float = -1e4,
    clim_dim: str = "number",
    ens_dim: str = "number",
) -> xr.DataArray | xr.Dataset:
    return utils.wrapper(array.sot, clim, ens, clim_dim, ens_dim, eps=eps, perc=perc)


def sot_unsorted(
    clim: xr.DataArray | xr.Dataset,
    ens: xr.DataArray | xr.Dataset,
    perc: int,
    eps: float = -1e4,
    clim_dim: str = "number",
    ens_dim: str = "number",
) -> xr.DataArray | xr.Dataset:
    return utils.wrapper(array.sot_unsorted, clim, ens, clim_dim, ens_dim, eps=eps, perc=perc)
