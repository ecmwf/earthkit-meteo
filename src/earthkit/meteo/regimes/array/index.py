# (C) Copyright 2025- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

from earthkit.utils.array import array_namespace


def project(fields, patterns, weights, patterns_coords=None):
    """Project onto the given regime patterns.

    Parameters
    ----------
    fields : array_like
        Input field(s) to project. The patterns are projected onto the trailing
        dimensions of the input fields.
    patterns : earthkit.meteo.regimes.Patterns
        Patterns to project on.
    weights : array_like
        Weights for the summation in the projection. Weights are normalised
        before application so the sum of weights over the domain equals 1. Must
        have shape of the patterns.
    patterns_coords : Mapping[str,Any], optional
        Keyword arguments for the pattern generation. E.g., a sequence of
        dates for date-modulated patterns. Must have the shape of `field`
        without the trailing dimensions onto which the patterns are projected.

    Returns
    -------
    array_like
        Results of the projection. Output fields have same shape as input field
        except that the dimensions reduced during the projection (i.e., the
        spatial dimensions of the patterns) are replaced by a regime dimension.
    """
    if patterns_coords is None:
        patterns_coords = {}
    ndim_field = len(patterns.shape)
    if fields.shape[-ndim_field:] != patterns.shape:
        raise ValueError(f"shape of input fields {fields.shape} incompatible with shape of patterns {patterns.shape}")

    if weights is None:
        # TODO generate area-based weights from grid of patterns with earthkit-geo
        # TODO make weights an optional argument with None default and document
        raise NotImplementedError("automatic generation of weights")
    if weights.shape != patterns.shape:
        raise ValueError(f"shape of weights {weights.shape} must match shape of patterns {patterns.shape}")
    weights = weights / weights.sum()

    fields = array_namespace(fields).expand_dims(fields, -ndim_field - 1)
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
