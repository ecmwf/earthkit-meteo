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


def cpf(
    clim: xr.DataArray | xr.Dataset,
    ens: xr.DataArray | xr.Dataset,
    sort_clim: bool = True,
    sort_ens: bool = True,
    epsilon: bool = None,
    symmetric: bool = False,
    clim_dim: str = "number",
    ens_dim: str = "number",
) -> xr.DataArray | xr.Dataset:
    return utils.wrapper(
        array.cpf,
        clim,
        ens,
        clim_dim,
        ens_dim,
        sort_clim=sort_clim,
        sort_ens=sort_ens,
        epsilon=epsilon,
        symmetric=symmetric,
    )
