
Version 0.4 Updates
/////////////////////////


Version 0.4.3
===============

Dependency updates
+++++++++++++++++++++

Ensure compatibility between earthkit components by restricting the following dependencies:

- ``earthkit-utils<0.2``


Version 0.4.1
===============

Array formats
++++++++++++++++++++++

Made almost all of the methods array format agnostic with support for Numpy, Torch and CuPy arrays/tensors as an input. The array backend is automatically detected from the input data type.

.. code-block:: python

    from earthkit.meteo import thermo

    # Example with Numpy array
    import numpy as np

    t = np.array([264.12, 261.45])
    p = np.array([850, 850]) * 100.0
    theta = thermo.potential_temperature(t, p)

    # Example with Torch tensor
    import torch

    t = torch.tensor([264.12, 261.45])
    p = torch.tensor([850.0, 850.0]) * 100.0
    theta = thermo.potential_temperature(t, p)


New features
+++++++++++++++++++++++

- Added the :py:mod:`earthkit.meteo.vertical` submodule
- Added the :py:meth:`earthkit.meteo.stats.array.value_to_return_period` and :py:meth:`earthkit.meteo.stats.array.return_period_to_value` methods. They are based on a Gumbel-distribution fit to the sample data (:pr:`29`). See the following notebook example:

    - :ref:`/how-tos/stats/return_period.ipynb`.

- Added :py:meth:`earthkit.meteo.score.array.correlation.pearson` to compute the Pearson correlation over fields.
- Implemented the symmetric Compute Crossing Point Forecast (CPF) method  :py:meth:`earthkit.meteo.extreme.array.cpf` (:pr:`28`).
- Added :py:meth:`earthkit.meteo.thermo.array.specific_gas_constant` to compute the specific gas constant for moist air
- Enabled :py:meth:`earthkit.meteo.solar.array.julian_day` to use timezone aware datetime objects as input (:pr:`41`).
- Added the ``nan_policy`` option to :py:meth:`earthkit.meteo.score.array.crps` to handle nans. The possible values are: "raise", "propagate", and "omit". The default is "propagate". (:pr:`45`)

New dependencies
+++++++++++++++++++++++

- earthkit-utils >= 0.0.1
