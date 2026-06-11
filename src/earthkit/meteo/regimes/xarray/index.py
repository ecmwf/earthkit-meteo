# (C) Copyright 2025- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import xarray as xr
from earthkit.utils.array import array_namespace

_PATTERN_DIM = "pattern"


def _labels_as_coord(patterns):
    values = patterns.xp.asarray(patterns.labels)
    return xr.DataArray(values, coords={_PATTERN_DIM: (_PATTERN_DIM, values)}, dims=[_PATTERN_DIM])


def _patterns_xr(patterns, reference_da, patterns_extra_coords):
    """Patterns evaluated for the given coords (if any) as xr.DataArrays.

    Parameters
    ----------
    reference_da : xr.DataArray
        Reference dataarray to take coordinates and dimension orders from.
    patterns_extra_coords : Mapping[str,str]
        Mapping of extra coordinates argument names (as given to .patterns)
        to DataArray coordinate names (as used in reference_da).

    Returns
    -------
    xarray.DataArray
    """
    import xarray as xr

    xp = patterns.xp
    # Extra coordinate dims, in order of reference dims
    extra_dims = [dim for dim in reference_da.dims if dim in patterns_extra_coords.values()]
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
    # coords and before grid coords
    assert _PATTERN_DIM not in dims
    dims.insert(-patterns.ndim, _PATTERN_DIM)
    coords[_PATTERN_DIM] = _labels_as_coord(patterns)
    # Cartesian product of coordinates for patterns generator
    extra_coords_arrs = dict(
        zip(
            extra_dims,
            xp.meshgrid(*(coords[dim] for dim in extra_dims), indexing="ij"),
        )
    )
    # Rearrange to match provided kwarg-coord mapping
    extra_coords = {kwarg: extra_coords_arrs[patterns_extra_coords[kwarg]] for kwarg in patterns_extra_coords}
    return xr.DataArray(patterns.patterns(**extra_coords), coords=coords, dims=dims)


def project(field, patterns, weights, **patterns_extra_coords):
    """Project onto the given patterns.

    Parameters
    ----------
    field : xarray.DataArray
        Input field(s) to project. The patterns are projected onto the trailing
        dimensions of the input fields.
    patterns : earthkit.meteo.regimes.Patterns
        Patterns to project on.
    weights : xarray.DataArray
        Weights for the summation in the projection. Weights are normalised
        before application so the sum of weights over the domain equals 1.
    **patterns_coords : dict[str,str], optional
        Mapping of coordinate names to keyword arguments of the pattern
        generation function. Only coordinates that are dimensions of `field`
        can be mapped.

    Returns
    -------
    xarray.DataArray
        The projection(s) for each pattern, with rightmost a ``"pattern"``
        dimension replacing the spatial dimension(s) reduced in the projection.
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
    patterns_da = _patterns_xr(patterns, field, patterns_extra_coords)
    return (field * patterns_da).weighted(weights).sum(dim=pattern_dims).rename("projection")


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
