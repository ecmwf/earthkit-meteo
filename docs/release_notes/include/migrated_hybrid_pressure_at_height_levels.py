import numpy as np

import earthkit.meteo.vertical as vertical
from earthkit.meteo.utils.sample import get_sample

# get hybrid (IFS model) level definition
A, B = vertical.hybrid_level_parameters(137, model="ifs")

# define surface pressures
sp = np.array([100000.0, 90000.0])

# get temperature and specific humidity profiles on hybrid levels (example data)
DATA = get_sample("vertical_hybrid_data")
t = DATA.t
q = DATA.q

# Option 1

# compute the pressure on full hybrid levels
p_full = vertical.pressure_on_hybrid_levels(A, B, sp, alpha_top="ifs", output="full")

# interpolate the pressure to geopotential height levels above the ground
# using linear interpolation
target_h = [10.0]
p_h = vertical.interpolate_hybrid_to_height_levels(
    p_full,
    target_h,
    t,
    q,
    0,
    A,
    B,
    sp,
    h_type="geopotential",
    h_reference="ground",
    interpolation="linear",
    alpha_top="ifs",
)

# Option 2

# compute the pressure on full hybrid levels
p_full, alpha, delta = vertical.pressure_on_hybrid_levels(
    A, B, sp, alpha_top="ifs", output=("full", "alpha", "delta")
)

z = vertical.relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta(t, q, alpha, delta)
h = vertical.geopotential_height_from_geopotential(z)

# interpolate the pressure to geopotential height levels above the ground
# using linear interpolation
target_h = [10.0]
p_h = vertical.interpolate_monotonic(
    p_full,
    h,
    target_h,
    interpolation="linear",
)
