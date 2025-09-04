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
        np.nan,
        -np.inf,
        np.nan,
        np.nan,
        # Edge-cases: observation
        np.nan,
        -np.inf,
        np.nan,
        np.nan,
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
        np.nan,
        1.0,
        np.nan,
        np.nan,
        # Edge-cases: observation
        np.nan,
        1.0,
        np.nan,
        np.nan,
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
        0.45454545,
        0.0,
        np.inf,
        -np.inf,
        # Edge-cases: observation
        2.2,
        np.inf,
        0,
        0,
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
        0.0,
        np.inf,
        np.nan,
        np.nan,
        # Edge-cases: observation
        np.inf,
        0.0,
        np.nan,
        np.nan,
    ],
]


v_ref_kge_components = [
    # KGE
    [
        1.0,
        -0.46476861,
        0.06826592,
        -0.40116796,
        -0.64678734,
        0.71900442,
        # Edge-cases: simulation
        np.nan,
        -0.00412373,
        np.nan,
        np.nan,
        # Edge-cases: observation
        np.nan,
        -np.inf,
        np.nan,
        np.nan,
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
        np.nan,
        1.0,
        np.nan,
        np.nan,
        # Edge-cases: observation
        np.nan,
        1.0,
        np.nan,
        np.nan,
    ],
    # alpha
    [
        1.0,
        0.18113070786957505,
        1.6906124944445566,
        0.5347125362695476,
        0.5644528931691898,
        0.7450688477363122,
        # Edge-cases: simulation
        0.0,
        0.9090909090909091,
        np.nan,
        np.nan,
        # Edge-cases: observation
        np.inf,
        1.0999999999999999,
        np.nan,
        np.nan,
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
        0.45454545,
        0.0,
        np.inf,
        -np.inf,
        # Edge-cases: observation
        2.2,
        np.inf,
        0,
        0,
    ],
]
