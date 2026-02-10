# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


def project(field, patterns, weights, **patterns_extra_coords):
    """Project onto the given regime patterns.

    Parameters
    ----------
    field : array_like
        Input field(s) to project. The regime patterns are projected onto the
        trailing dimensions of the input fields.
    patterns : earthkit.meteo.regimes.RegimePatterns
        Regime patterns.
    weights : array_like
        Weights for the summation in the projection. Weights are normalised
        before application so the sum of weights over the domain equals 1. Must
        have shape of the regime patterns.
    **patterns_coords : dict[str, Any], optional
        Keyword arguments for the pattern generation. E.g., a sequence of
        dates for date-modulated regime patterns. Must have shape of input
        fields without the trailing dimensions onto which the patterns are
        projected.

    Returns
    -------
    dict[str, array_like]
        Results of the projection for each regime.
    """
    ndim_field = len(patterns.shape)
    if field.shape[-ndim_field:] != patterns.shape:
        raise ValueError(
            f"shape of input fields {field.shape} incompatible with shape of regime patterns {patterns.shape}"
        )

    if weights is None:
        # TODO generate area-based weights from grid of patterns with earthkit-geo
        # TODO make weights an optional argument with None default and document
        raise NotImplementedError("automatic generation of weights")
    if weights.shape != patterns.shape:
        raise ValueError(f"shape of weights {weights.shape} must match shape of patterns {patterns.shape}")
    weights = weights / weights.sum()

    # Project onto each regime pattern
    sum_axes = tuple(range(-ndim_field, 0, 1))
    return {
        regime: (field * pattern * weights).sum(axis=sum_axes)
        for regime, pattern in patterns.patterns(**patterns_extra_coords).items()
    }


def standardise(projections, mean, std):
    """Regime index by standardisation of regime projections.

    Convenience function to work with dictionaries.

    Parameters
    ----------
    projections : dict[str, array_like]
        Projections onto regime patterns.
    mean : dict[str, array_like]
    std : dict[str, array_like]

    Returns
    -------
    dict[str, array_like]
        ``(projection - mean) / std`` for each regime
    """
    return {regime: (proj - mean[regime]) / std[regime] for regime, proj in projections.items()}
