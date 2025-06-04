Welcome to earthkit-meteo's documentation
======================================================

|Static Badge| |image1| |License: Apache 2.0| |Latest
Release|

.. |Static Badge| image:: https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE/foundation_badge.svg
   :target: https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE
.. |image1| image:: https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/incubating_badge.svg
   :target: https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity
.. |License: Apache 2.0| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/apache-2-0
.. |Latest Release| image:: https://img.shields.io/github/v/release/ecmwf/earthkit-meteo?color=blue&label=Release&style=flat-square
   :target: https://github.com/ecmwf/earthkit-meteo/releases


.. important::

    This software is **Incubating** and subject to ECMWF's guidelines on `Software Maturity <https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity>`_.


**earthkit-meteo** is a Python package providing meteorological computations using array input (Numpy, Torch and CuPy) and output. It is part of the :xref:`earthkit` ecosystem.


Quick start
-----------

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



.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 1
   :caption: Documentation

   _api/meteo/index
   references.rst
   development

.. toctree::
   :maxdepth: 1
   :caption: Installation

   install
   release_notes/index
   licence


.. toctree::
   :maxdepth: 1
   :caption: Projects

   earthkit <https://earthkit.readthedocs.io/en/latest>


Indices and tables
==================

* :ref:`genindex`

.. * :ref:`modindex`
.. * :ref:`search`
