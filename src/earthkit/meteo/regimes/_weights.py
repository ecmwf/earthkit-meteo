# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.


def generate_area_weights(grid, xp):
    """Generate area-based weights for the given grid."""
    if grid.type in {"regular_ll", "regular-ll"}:
        lats, _ = xp.asarray(grid.to_latlons())
        return xp.cos(xp.deg2rad(lats)).reshape(grid.shape)
    raise NotImplementedError(f"weights generation not available for grid type {grid.type}")


def prepare_normalised_weights(patterns, weights=None):
    """Normalize and verify given weights or generate normalised weights for the patterns.

    Parameters
    ----------
    patterns : earthkit.meteo.regimes.Patterns
        The patterns for which to generate weights for or verify the given
        weights against.
    weights : array_like | None
        The weights to verify and normalise.

    Returns
    -------
    array_like
    """
    xp = patterns.xp
    if weights is None:
        weights = generate_area_weights(patterns.grid, xp)
    else:
        weights = xp.asarray(weights)
    if weights.shape != patterns.shape:
        raise ValueError(f"shape {weights.shape} of weights must match shape {patterns.shape} of patterns {patterns!r}")
    return weights / weights.sum()
