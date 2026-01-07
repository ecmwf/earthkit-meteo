import numpy as np

import earthkit.meteo.vertical as vertical
from earthkit.meteo.utils.testing import hybrid_level_test_data

# get hybrid (IFS model) level definition
A, B = vertical.hybrid_level_parameters(137, model="ifs")

# define surface pressures
sp = np.array([100000.0, 90000.0])

# get temperature and specific humidity profiles on hybrid levels (example data)
DATA = hybrid_level_test_data()
t = DATA.t
q = DATA.q

# compute the relative geopotential thickness
z_thickness = vertical.relative_geopotential_thickness_on_hybrid_levels(t, q, A, B, sp, alpha_top="ifs")
