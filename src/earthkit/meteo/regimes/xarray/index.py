# (C) Copyright 2025- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import xarray as xr


def project(field, patterns, weights, **patterns_extra_coords):
    """Project onto the given patterns.

    Parameters
    ----------
    field : xarray.Dataarray
        Input field(s) to project. The patterns are projected onto the trailing
        dimensions of the input fields.
    patterns : earthkit.meteo.regimes.Patterns
        Patterns to project on.
    weights : xarray.Dataarray
        Weights for the summation in the projection. Weights are normalised
        before application so the sum of weights over the domain equals 1.
    **patterns_coords : dict[str,str], optional
        Mapping of coordinate names to keyword arguments of the pattern
        generation function. Only coordinates that are dimensions of `field`
        can be mapped.

    Returns
    -------
    xarray.DataArray
        The projection(s) for each pattern.
    """
    # Dimensions of a single pattern, assumed to be the trailing dimensions
    field_trailing_shape = field.shape[-patterns.ndim :]
    if field_trailing_shape != patterns.shape:
        raise ValueError(
            "trailing dimensions of input field must match shape of patterns: "
            f"expected {patterns.shape}, got {field_trailing_shape}"
        )
    pattern_dims = field.dims[-patterns.ndim :]
    # Normalise weights so they sum to zero over the pattern domain and
    # compensate for weights that don't have all pattern dimensions
    if weights is None:
        raise NotImplementedError("automatic generation of weights")
    if set(weights.dims) - set(pattern_dims):
        raise ValueError("weight must only be specified over pattern dimensions")
    weights = weights / weights.sum() * weights.size / patterns.size
    # Matching the behaviour of array.project, introduce the regime dimension
    # as a new outermost dimension
    return xr.concat(
        [
            (field * pattern).weighted(weights).sum(dim=pattern_dims).assign_coords({"pattern": label})
            for label, pattern in patterns._patterns_iterxr(field, patterns_extra_coords)
        ],
        dim="pattern",
    ).rename("projection")


def regime_index(projections, mean, std):
    """Regime index by standardisation of projections onto patterns.

    Parameters
    ----------
    projections : xarray.Dataarray
        Projections onto regime patterns.
    mean : xarray.Dataarray
    std : xarray.Dataarray

    Returns
    -------
    xarray.Dataarray
        ``(projection - mean) / std``
    """
    return ((projections - mean) / std).rename("IWR")
