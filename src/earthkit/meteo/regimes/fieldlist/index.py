# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import earthkit.data as ekd
import pandas as pd


def project(field, patterns, weights, **patterns_extra_coords):
    """Project onto the given patterns.

    Parameters
    ----------
    field : earthkit.data.FieldList | earthkit.data.Field
        Input fields whose values the patterns are projected onto.
    patterns : earthkit.meteo.regimes.Patterns
        Patterns to project on.
    weights : array_like
        Weights for the summation in the projection. Weights are normalised
        before application so the sum of weights over the domain equals 1. Must
        have the shape of the patterns.
    **patterns_extra_coords : dict[str, array_like], optional
        Mapping of field metadata keys to keyword arguments of the pattern
        generator.

    Returns
    -------
    pandas.DataFrame
        The projection(s) for each pattern. One column per pattern, one row
        per field (same order as input fields).
    """
    xp = patterns.xp

    if weights is None:
        # TODO generate area-based weights from grid of patterns with earthkit-geo
        raise NotImplementedError("automatic generation of weights")
    weights = xp.asarray(weights)
    if weights.shape != patterns.shape:
        raise ValueError(f"shape of weights {weights.shape} must match shape of patterns {patterns.shape}")
    weights = weights / weights.sum()

    if not isinstance(field, ekd.FieldList):
        field = field.to_fieldlist()
    # Project field-by-field to keep peak memory usage down
    proj = {label: xp.empty(len(field)) for label in patterns.labels}
    for i, fld in enumerate(field):
        # TODO verify grid compatibility and regrid with earthkit.geo on mismatch
        values = fld.data(keys="value", flatten=False)
        # Extract extra coordinates required for the pattern generation
        coords = {kwarg: fld.get(coord) for kwarg, coord in patterns_extra_coords.items()}
        for label, pattern in patterns.patterns(**coords).items():
            proj[label][i] = (values * pattern * weights).sum()

    return pd.DataFrame.from_dict(proj)
