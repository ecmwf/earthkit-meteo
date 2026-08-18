# (C) Copyright 2025- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

from earthkit.utils.array import array_namespace

from .._weights import prepare_normalised_weights


def project(fields, patterns, weights=None, patterns_coords=None):
    """Project onto the given regime patterns.

    Parameters
    ----------
    fields : array_like
        Input field(s) to project. The patterns are projected onto the trailing
        dimensions of the input fields.
    patterns : earthkit.meteo.regimes.Patterns
        Patterns to project on.
    weights : array_like, optional
        Weights for the summation in the projection. Weights are normalised
        before application so the sum of weights over the domain equals 1. Must
        have shape of the patterns. If no weights are specified, area-based
        weights are generated from the cosine of latitude of the patterns grid.
    patterns_coords : Mapping[str,Any], optional
        Keyword arguments for the pattern generation. E.g., a sequence of
        dates for date-modulated patterns. Each value must have the shape of
        `fields` without the trailing dimensions onto which the patterns are
        projected.

    Returns
    -------
    array_like
        Results of the projection. Output fields have same shape as input fields
        except that the dimensions reduced during the projection (i.e., the
        spatial dimensions of the patterns) are replaced by a regime dimension.
    """
    if patterns_coords is None:
        patterns_coords = {}
    ndim_field = len(patterns.shape)
    fields = array_namespace(fields).expand_dims(fields, -ndim_field - 1)
    if fields.shape[-ndim_field:] != patterns.shape:
        raise ValueError(f"shape of input fields {fields.shape} incompatible with shape of patterns {patterns.shape}")
    weights = prepare_normalised_weights(patterns, weights=weights)
    sum_axes = tuple(range(-ndim_field, 0, 1))
    return (fields * patterns.patterns(**patterns_coords) * weights).sum(axis=sum_axes)


def regime_index(projections, mean, std):
    """Regime index by standardisation of projections onto patterns.

    Parameters
    ----------
    projections : array_like
        Projections onto regime patterns.
    mean : array_like
    std : array_like

    Returns
    -------
    array_like
        ``(projection - mean) / std`` for each regime
    """
    return (projections - mean) / std
