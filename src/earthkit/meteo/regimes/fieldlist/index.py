# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

import earthkit.data as ekd
import earthkit.geo as ekg

from .. import array as regimes_array
from .._weights import prepare_normalised_weights


def project(fields, patterns, weights=None, patterns_coords=None, regrid_to_pattern=True):
    """Project onto the given patterns.

    Parameters
    ----------
    fields : earthkit.data.FieldList | earthkit.data.Field
        Input fields whose values the patterns are projected onto.
    patterns : earthkit.meteo.regimes.Patterns
        Patterns to project on.
    weights : array_like, optional
        Weights for the summation in the projection. Weights are normalised
        before application so the sum of weights over the domain equals 1. Must
        have the shape of the patterns. If no weights are specified, area-based
        weights are generated from the patterns grid.
    patterns_coords : Mapping[str,str], optional
        Mapping of field metadata keys to keyword arguments of the pattern
        generator.
    regrid_to_pattern : bool
        Allow regridding of input fields to match the pattern grid. Enabled by
        default.

    Returns
    -------
    array_like
        The projection(s) for each pattern. One row per field, one column per
        pattern. Same order as input fields and pattern labels, respectively.
    """
    if patterns_coords is None:
        patterns_coords = {}
    weights = prepare_normalised_weights(patterns, weights)
    if not isinstance(fields, ekd.FieldList):
        fields = fields.to_fieldlist()
    proj = []
    for field in fields:
        # Automatic regridding, also covers cropping to pattern area
        if regrid_to_pattern and field.get("geography.grid") != patterns.grid:
            try:
                field = ekg.regrid(field, out_grid=patterns.grid)
            except RuntimeError as e:
                raise RuntimeError(
                    f"regrid_to_pattern=True but regridding to pattern grid {patterns.grid!r} failed for {fields!r}"
                ) from e
        values = field.data(keys="value", flatten=False)
        # Extract extra coordinates required for the pattern generation
        coords = {kwarg: field.get(coord) for kwarg, coord in patterns_coords.items()}
        proj.append(regimes_array.project(values, patterns, weights, patterns_coords=coords))
    return patterns.xp.asarray(proj)
