# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from earthkit.utils.decorators import dispatch as dispatch
from earthkit.utils.decorators import fieldlist_ufunc as fieldlist_ufunc
from earthkit.utils.decorators import xarray_ufunc as xarray_ufunc


def get_dim_from_defaults(da, dim: str | None, dim_names: tuple[str, ...]) -> str | None:
    """Get dimension name from defaults if not provided."""
    if dim is not None:
        return dim
    for name in dim_names:
        if name in da.dims:
            return name
    return None
