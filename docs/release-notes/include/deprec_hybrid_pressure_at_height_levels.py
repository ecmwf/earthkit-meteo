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

# compute the pressure on geopotential height levels above the
# ground using linear interpolation
h_target = 10.0
p = vertical.pressure_at_height_levels(h_target, t, q, sp, A, B, alpha_top="ifs")
