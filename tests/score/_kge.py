import numpy as np

sim = [
    # Normal cases
    [0.0, 1.0, 2.0],
    [0.05, 0.48, 0.77],
    [0.06, 0.66, 0.45],
    [0.21, 0.05, 0.16],
    [0.61, 0.8, 0.89],
    [0.53, 0.14, 0.3],
    # Edge-cases: simulation
    [1.0, 1.0, 1.0],  # zero variance
    [-1.0, 0.0, 1.0],  # zero mean
    [1.0, 2, np.inf],  # Inf
    [1.0, 2, -np.inf],  # -Inf
    # Edge-cases: observation
    [1.1, 2.2, 3.3],
    [1.1, 2.2, 3.3],
    [1.1, 2.2, 3.3],
    [1.1, 2.2, 3.3],
]

obs = [
    [0.0, 1.0, 2.0],
    [-5.0, -1.0, -3.0],
    [0.39, 0.58, 0.22],
    [0.44, 0.43, 0.17],
    [0.52, 0.7, 0.2],
    [0.65, 0.13, 0.32],
    # Edge-cases: simulation
    [1.1, 2.2, 3.3],
    [1.1, 2.2, 3.3],
    [1.1, 2.2, 3.3],
    [1.1, 2.2, 3.3],
    # Edge-cases: observation
    [1.0, 1.0, 1.0],  # zero variance
    [-1.0, 0.0, 1.0],  # zero mean
    [1.0, 2, np.inf],  # Inf
    [1.0, 2, -np.inf],  # -Inf
]

v_ref_kge_prime_components = [
    # KGE'
    [
        1.0000000000,
        -1.5603582218,
        0.0466479764,
        -0.3608040612,
        -0.7165881091,
        0.8050195736,
        # Edge-cases: simulation
        np.nan,  # zero variance
        -np.inf,  # zero mean
        np.nan,  # inf
        np.nan,  # -inf
        # Edge-cases: observation
        np.nan,  # zero variance
        -np.inf,  # zero mean
        np.nan,  # inf
        np.nan,  # -inf
    ],
    # rho
    [
        1.0000000000,
        0.5934940644,
        0.3747797925,
        -0.1795676428,
        -0.4622436417,
        0.9987024321,
        # Edge-cases: simulation
        np.nan,  # zero variance
        1.0,  # zero mean
        np.nan,  # inf
        np.nan,  # -inf
        # Edge-cases: observation
        np.nan,  # zero variance
        1.0,  # zero mean
        np.nan,  # inf
        np.nan,  # -inf
    ],
    # beta
    [
        1.0000000000,
        -0.1444444444,
        0.9831932773,
        0.4038461538,
        1.6197183099,
        0.8818181818,
        # Edge-cases: simulation
        0.45454545,  # zero variance
        0.0,  # zero mean
        np.inf,  # inf
        -np.inf,  # -inf
        # Edge-cases: observation
        2.2,  # zero variance
        np.inf,  # zero mean
        0,  # inf
        0,  # -inf
    ],
    # gamma
    [
        1.0000000000,
        -1.2539818237,
        1.7195118533,
        1.3240500898,
        0.3484883080,
        0.8449234356,
        # Edge-cases: simulation
        0.0,  # zero variance
        np.inf,  # zero mean
        np.nan,  # inf
        np.nan,  # -inf
        # Edge-cases: observation
        np.inf,  # zero variance
        0.0,  # zero mean
        np.nan,  # inf
        np.nan,  # -inf
    ],
]
