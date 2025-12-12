# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np


def project(field, patterns, weights=None, **patterns_kwargs):
    """Project onto the given regime patterns.

    Parameters
    ----------
    field : array_like
        Input field(s) to project.
    patterns : earthkit.meteo.regimes.RegimePatterns
        Regime patterns.
    weights : None, array_like, "uniform", optional
        Weights for the summation over the spatial dimensions.
    **patterns_kwargs : dict[str, Any], optional
        Keyword argumenents for the pattern generation. E.g., a sequence of
        dates for date-modulated regime patterns.

    Returns
    -------
    dict[str, array_like]
        Results of the projection for each regime.
    """
    ndim_field = len(patterns.shape)
    assert field.shape[-ndim_field:] == patterns.shape

    ps = patterns.patterns(**patterns_kwargs)

    if weights is None:
        # TODO generate area-based weights from grid of patterns with earthkit-geo
        raise NotImplementedError
    assert weights.shape == patterns.shape
    weights = weights / np.sum(weights)

    # Project onto each regime pattern
    sum_axes = tuple(range(-ndim_field, 0, 1))
    return {regime: np.sum(field * pattern * weights, axis=sum_axes) for regime, pattern in ps.items()}


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
