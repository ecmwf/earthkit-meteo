# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""Vertical computation functions operating on numpy arrays."""

from .hybrid import hybrid_level_parameters
from .vertical import (
    geometric_height_from_geopotential,
    geometric_height_from_geopotential_height,
    geopotential_from_geometric_height,
    geopotential_from_geopotential_height,
    geopotential_height_from_geometric_height,
    geopotential_height_from_geopotential,
    geopotential_on_hybrid_levels,
    height_on_hybrid_levels,
    interpolate_hybrid_to_height_levels,
    interpolate_hybrid_to_pressure_levels,
    interpolate_monotonic,
    interpolate_pressure_to_height_levels,
    pressure_at_height_levels,
    pressure_at_model_levels,
    pressure_on_hybrid_levels,
    relative_geopotential_thickness,
    relative_geopotential_thickness_on_hybrid_levels,
    relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta,
)

__all__ = [
    "hybrid_level_parameters",
    "geometric_height_from_geopotential",
    "geometric_height_from_geopotential_height",
    "geopotential_from_geometric_height",
    "geopotential_from_geopotential_height",
    "geopotential_height_from_geometric_height",
    "geopotential_height_from_geopotential",
    "geopotential_on_hybrid_levels",
    "height_on_hybrid_levels",
    "interpolate_hybrid_to_height_levels",
    "interpolate_hybrid_to_pressure_levels",
    "interpolate_monotonic",
    "interpolate_pressure_to_height_levels",
    "pressure_at_height_levels",
    "pressure_at_model_levels",
    "pressure_on_hybrid_levels",
    "relative_geopotential_thickness",
    "relative_geopotential_thickness_on_hybrid_levels",
    "relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta",
]
