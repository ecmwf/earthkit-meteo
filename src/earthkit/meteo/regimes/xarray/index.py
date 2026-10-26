# (C) Copyright 2025- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

from collections.abc import Sequence

import xarray as xr
from earthkit.utils.array import array_namespace


def _patterns_xr(patterns, reference_da, patterns_coords, patterns_dim="pattern"):
    """Patterns evaluated for the given coords (if any) as xr.DataArrays.

    Parameters
    ----------
    patterns : earthkit.meteo.regimes.Patterns
    reference_da : xr.DataArray
        Reference dataarray to take coordinates and dimension orders from.
    patterns_coords : Mapping[str,str]
    pattern_dim : str, optional
        Name of the pattern dimension that replaces the grid dimensions.

    Returns
    -------
    xarray.DataArray
    """
    if patterns_dim in reference_da.dims:
        raise ValueError("pattern dimension '{patterns_dim}' already exists")
    xp = patterns.xp
    # Extra coordinate dims, in order of reference dims
    extra_dims = [dim for dim in reference_da.dims if dim in patterns_coords.values()]
    # Output dimensions and coordinates of the patterns
    dims = [*extra_dims, *reference_da.dims[-patterns.ndim :]]
    coords = {dim: reference_da.coords[dim] for dim in dims}
    # Lazy and chunked pattern generation: if the reference dataset is
    # chunked, transfer its chunking to the coordinates and use the chunk-
    # enabled array namespace in the next step
    if reference_da.chunksizes:
        xp = array_namespace(reference_da.data)
        coords = {dim: xp.asarray(values).rechunk(reference_da.chunksizes[dim]) for dim, values in coords.items()}
    # Regime pattern coordinate based on pattern labels: insert after extra
    # coords and before grid coords. This leaves it as the innermost dimension
    # after the projection.
    dims.insert(len(dims) - patterns.ndim, patterns_dim)
    coords[patterns_dim] = xr.DataArray(xp.asarray(patterns.labels), dims=[patterns_dim])
    # Cartesian product of coordinates for patterns generator
    extra_coords_arrs = dict(
        zip(
            extra_dims,
            xp.meshgrid(*(coords[dim] for dim in extra_dims), indexing="ij"),
        )
    )
    # Rearrange to match provided kwarg-coord mapping
    extra_coords = {kwarg: extra_coords_arrs[coord] for kwarg, coord in patterns_coords.items()}
    return xr.DataArray(patterns.patterns(**extra_coords), coords=coords, dims=dims)


def project(fields, patterns, weights, patterns_coords=None):
    """Project onto the given patterns.

    Parameters
    ----------
    fields : xarray.DataArray
        Input field(s) to project. The patterns are projected onto the trailing
        dimensions of the input fields.
    patterns : earthkit.meteo.regimes.Patterns
        Patterns to project on.
    weights : xarray.DataArray
        Weights for the summation in the projection. Weights are normalised
        before application so the sum of weights over the domain equals 1.
    patterns_coords : Mapping[str,str] | Sequence[str], optional
        Mapping of coordinate names to keyword arguments of the pattern
        generation function. If a sequence is given, argument and associated
        coordinate names are assumed to be identical. Only coordinates that are
        dimensions of `fields` can be mapped.

    Returns
    -------
    xarray.DataArray
        The projection(s) for each pattern, with rightmost a ``"pattern"``
        dimension replacing the spatial dimension(s) reduced in the projection.
    """
    if patterns_coords is None:
        patterns_coords = {}
    elif isinstance(patterns_coords, Sequence):
        patterns_coords = {coord: coord for coord in patterns_coords}

    # Dimensions of a single pattern, assumed to be the trailing dimensions
    field_trailing_shape = fields.shape[-patterns.ndim :]
    if field_trailing_shape != patterns.shape:
        raise ValueError(
            "trailing dimensions of input field must match shape of patterns: "
            f"expected {patterns.shape}, got {field_trailing_shape}"
        )
    pattern_dims = fields.dims[-patterns.ndim :]

    # Normalise weights so they sum to one over the pattern domain and
    # compensate for weights that don't have all pattern dimensions
    if weights is None:
        raise NotImplementedError("automatic generation of weights")
    if set(weights.dims) - set(pattern_dims):
        raise ValueError("weight must only be specified over pattern dimensions")
    weights = weights / weights.sum() * weights.size / patterns.size

    patterns_da = _patterns_xr(patterns, fields, patterns_coords)
    return (fields * patterns_da).weighted(weights).sum(dim=pattern_dims).rename("projection")


def regime_index(projections, mean, std):
    """Regime index by standardisation of projections onto patterns.

    Parameters
    ----------
    projections : xarray.DataArray
        Projections onto regime patterns.
    mean : xarray.DataArray
    std : xarray.DataArray

    Returns
    -------
    xarray.DataArray
        ``(projection - mean) / std``
    """
    return ((projections - mean) / std).rename("IWR")
