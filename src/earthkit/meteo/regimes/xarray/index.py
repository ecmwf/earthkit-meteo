# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import xarray as xr


def project(field, patterns, weights, **patterns_extra_coords) -> xr.DataArray:
    # Dimensions of a single pattern, assumed to be the trailing dimensions
    assert field.shape[-patterns.ndim :] == patterns.shape
    pattern_dims = field.dims[-patterns.ndim :]
    # Matching the behaviour of array.project, introduce the regime dimension
    # as a new outermost dimension. apply_ufunc can't do this.
    weighted_field = field * weights
    return xr.concat(
        [
            (weighted_field * pattern).sum(dim=pattern_dims).assign_coords({"regime": regime})
            for regime, pattern in patterns._patterns_iterxr(field, patterns_extra_coords)
        ],
        dim="regime",
    )


def standardise(projections, mean, std):
    return (projections - mean) / std
