# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import earthkit.data as ekd
import pandas as pd

from .. import array as regimes_array
from .._weights import prepare_normalised_weights


def project(field, patterns, weights=None, patterns_coords=None):
    """Project onto the given patterns.

    Parameters
    ----------
    field : earthkit.data.FieldList | earthkit.data.Field
        Input fields whose values the patterns are projected onto.
    patterns : earthkit.meteo.regimes.Patterns
        Patterns to project on.
    weights : array_like, optional
        Weights for the summation in the projection. Weights are normalised
        before application so the sum of weights over the domain equals 1. Must
        have the shape of the patterns. If no weights are specified, area-based
        weights are generated from the cosine of latitude of the patterns grid.
    patterns_coords : Mapping[str,str], optional
        Mapping of field metadata keys to keyword arguments of the pattern
        generator.

    Returns
    -------
    pandas.DataFrame
        The projection(s) for each pattern. One column per pattern, one row
        per field (same order as input fields).
    """
    if patterns_coords is None:
        patterns_coords = {}
    weights = prepare_normalised_weights(weights, patterns)
    if not isinstance(field, ekd.FieldList):
        field = field.to_fieldlist()
    # Project field-by-field to keep peak memory usage down
    proj = []
    for fld in field:
        values = fld.data(keys="value", flatten=False)
        # Extract extra coordinates required for the pattern generation
        coords = {kwarg: fld.get(coord) for kwarg, coord in patterns_coords.items()}
        proj.append(regimes_array.project(values, patterns, weights, patterns_coords=coords))
    return pd.DataFrame.from_records(proj, columns=patterns.labels)
