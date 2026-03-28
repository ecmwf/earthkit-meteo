import numpy as np

import earthkit.meteo.vertical as vertical

# get hybrid (IFS model) level definition
A, B = vertical.hybrid_level_parameters(137, model="ifs")

# define surface pressures
sp = np.array([100000.0, 90000.0])

p_full, p_half, delta, alpha = vertical.pressure_at_model_levels(A, B, sp, alpha_top="ifs")
