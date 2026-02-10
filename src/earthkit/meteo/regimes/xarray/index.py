# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import xarray as xr


def project(field, patterns, weights, **patterns_extra_coords) -> xr.DataArray:
    """Project onto the given regime patterns.

    Parameters
    ----------
    field : xarray.Dataarray
        Input field(s) to project. The regime patterns are projected onto the
        trailing dimensions of the input fields.
    patterns : earthkit.meteo.regimes.RegimePatterns
        Regime patterns.
    weights : xarray.Dataarray
        Weights for the summation in the projection. Weights are normalised
        before application so the sum of weights over the domain equals 1.
    **patterns_coords : dict[str, str], optional
        Mapping of coordinate names to keyword arguments of the pattern
        generation. The coordinates must be dimensions of `field`.

    Returns
    -------
    xarray.Dataarray
        Results of the projection for each regime.
    """
    # Dimensions of a single pattern, assumed to be the trailing dimensions
    assert field.shape[-patterns.ndim :] == patterns.shape
    pattern_dims = field.dims[-patterns.ndim :]
    # Normalise weights so they sum to zero over the pattern domain and
    # compensate for weights that don't have all pattern dimensions
    if weights is None:
        raise NotImplementedError("automatic generation of weights")
    weights = weights / weights.sum() * weights.size / patterns.size
    # Matching the behaviour of array.project, introduce the regime dimension
    # as a new outermost dimension
    return xr.concat(
        [
            (field * pattern).weighted(weights).sum(dim=pattern_dims).assign_coords({"regime": regime})
            for regime, pattern in patterns._patterns_iterxr(field, patterns_extra_coords)
        ],
        dim="regime",
    )


def standardise(projections, mean, std):
    """Regime index by standardisation of regime projections.

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
    return (projections - mean) / std
