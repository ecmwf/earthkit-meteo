Installation and Getting Started
================================

Installing from PyPi
--------------------

Install **earthkit-meteo** with python3 (>= 3.10) and ``pip`` as follows:

.. code-block:: bash

    python3 -m pip install earthkit-meteo



Import and use
--------------


.. code-block:: python

    from earthkit.meteo import thermo

    # using Numpy arrays
    import numpy as np

    t = np.array([264.12, 261.45])  # Kelvins
    p = np.array([850, 850]) * 100.0  # Pascals
    theta = thermo.potential_temperature(t, p)

    # using Torch tensors
    import torch

    t = torch.tensor([264.12, 261.45])  # Kelvins
    p = torch.tensor([850.0, 850.0]) * 100.0  # Pascals
    theta = thermo.potential_temperature(t, p)
