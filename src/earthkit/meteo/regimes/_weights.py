# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.


def generate_area_weights(grid, xp):
    if grid.type == "regular-ll":
        lats, _ = xp.asarray(grid.to_latlons())
        return xp.cos(xp.deg2rad(lats)).reshape(grid.shape)
    raise NotImplementedError("weights generation not available for grid type {grid.type}")


def prepare_normalised_weights(weights, patterns):
    xp = patterns.xp
    if weights is None:
        weights = generate_area_weights(patterns.grid, xp)
    else:
        weights = xp.asarray(weights)
    if weights.shape != patterns.shape:
        raise ValueError(f"shape of weights {weights.shape} must match shape of patterns {patterns.shape}")
    return weights / weights.sum()
