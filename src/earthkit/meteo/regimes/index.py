# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

from ..utils.decorators import dispatch

__all__ = ["project", "regime_index"]


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
        The projection(s) for each pattern, with "pattern" as a new leftmost
        dimension and all dimensions of field following except for the
        dimensions reduced in the projection (i.e., the spatial dimensions of
        the patterns are missing on the right).
    """
    return dispatch(project, field, patterns, weights, **patterns_extra_coords)


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
    return dispatch(regime_index, projections, mean, std)
