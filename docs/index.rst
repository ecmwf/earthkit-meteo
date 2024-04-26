Welcome to earthkit-meteo's documentation
======================================================

.. warning::

    This project is **BETA** and will be **Experimental** for the foreseeable future. Interfaces and functionality are likely to change, and the project itself may be scrapped. **DO NOT** use this software in any project/software that is operational.


**earthkit-meteo** is a Python package providing meteorological computations using **numpy** input and output.

.. code-block:: python

    from earthkit.meteo import thermo
    import numpy as np

    t = np.array([264.12, 261.45])  # Kelvins
    p = np.array([850, 850]) * 100.0  # Pascals

    theta = thermo.potential_temperature(t, p)


.. .. toctree::
..    :maxdepth: 1
..    :caption: Examples
..    :titlesonly:

..    examples

.. toctree::
   :maxdepth: 1
   :caption: Documentation

   _api/meteo/index
   references.rst

.. toctree::
   :maxdepth: 1
   :caption: Installation

   install
   release_notes/index
   development
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
