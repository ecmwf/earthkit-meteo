# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

"""
Vertical computation functions.

The API is organised in layers:

- Core numerical routines live in the ``array`` submodule.
- Functions exposed from this module provide the high-level entry points for the
  vertical API.

For xarray interpolation workflows, see :mod:`earthkit.meteo.vertical.interpolation`.
"""

from . import array, xarray
from .interpolation import (
    interpolate_monotonic,
    interpolate_sleve_to_coord_levels,
    interpolate_sleve_to_theta_levels,
    interpolate_to_pressure_levels,
)
from .vertical import (
    geometric_height_from_geopotential,
    geometric_height_from_geopotential_height,
    geopotential_from_geometric_height,
    geopotential_from_geopotential_height,
    geopotential_height_from_geometric_height,
    geopotential_height_from_geopotential,
    geopotential_on_hybrid_levels,
    height_on_hybrid_levels,
    hybrid_level_parameters,
    interpolate_hybrid_to_height_levels,
    interpolate_hybrid_to_pressure_levels,
    interpolate_pressure_to_height_levels,
    pressure_at_height_levels,
    pressure_at_model_levels,
    pressure_on_hybrid_levels,
    relative_geopotential_thickness,
    relative_geopotential_thickness_on_hybrid_levels,
    relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta,
)

__all__ = [
    "interpolate_monotonic",
    "interpolate_sleve_to_coord_levels",
    "interpolate_sleve_to_theta_levels",
    "interpolate_to_pressure_levels",
    "geometric_height_from_geopotential",
    "geometric_height_from_geopotential_height",
    "geopotential_from_geometric_height",
    "geopotential_from_geopotential_height",
    "geopotential_height_from_geometric_height",
    "geopotential_height_from_geopotential",
    "geopotential_on_hybrid_levels",
    "height_on_hybrid_levels",
    "hybrid_level_parameters",
    "interpolate_hybrid_to_height_levels",
    "interpolate_hybrid_to_pressure_levels",
    "interpolate_pressure_to_height_levels",
    "pressure_at_height_levels",
    "pressure_at_model_levels",
    "pressure_on_hybrid_levels",
    "relative_geopotential_thickness",
    "relative_geopotential_thickness_on_hybrid_levels",
    "relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta",
    "array",
    "xarray",
]
