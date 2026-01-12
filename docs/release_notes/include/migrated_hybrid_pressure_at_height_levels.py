import numpy as np

import earthkit.meteo.vertical as vertical
from earthkit.meteo.utils.sample import get_sample

# get hybrid (IFS model) level definition
A, B = vertical.hybrid_level_parameters(137, model="ifs")

# define surface pressures
sp = np.array([100000.0, 90000.0])

# compute the pressure on full hybrid levels
p_full = vertical.pressure_on_hybrid_levels(A, B, sp, alpha_top="ifs", output="full")

# get temperature and specific humidity profiles on hybrid levels (example data)
DATA = get_sample("vertical_hybrid_data")
t = DATA.t
q = DATA.q

# interpolate the pressure to geopotential height levels above the ground
# using linear interpolation
target_h = [10.0]
p_h = vertical.interpolate_hybrid_to_height_levels(
    p_full,
    t,
    q,
    0,
    A,
    B,
    sp,
    target_h,
    h_type="geopotential",
    h_reference="ground",
    interpolation="linear",
    alpha_top="ifs",
)
