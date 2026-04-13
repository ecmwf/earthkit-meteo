import numpy as np

import earthkit.meteo.vertical as vertical
from earthkit.meteo.utils.sample import get_sample

# get hybrid (IFS model) level definition
A, B = vertical.hybrid_level_parameters(137, model="ifs")

# define surface pressures
sp = np.array([100000.0, 90000.0])

# compute alpha and delta
_, _, delta, alpha = vertical.pressure_at_model_levels(A, B, sp, alpha_top="ifs")

# get temperature and specific humidity profiles on hybrid levels (example data)
DATA = get_sample("vertical_hybrid_data")
t = DATA.t
q = DATA.q

# compute the relative geopotential thickness
z_thickness = vertical.relative_geopotential_thickness(alpha, delta, t, q)
