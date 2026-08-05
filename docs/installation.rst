.. _install:

Installation
============

Installing from PyPI
--------------------

Minimal installation
+++++++++++++++++++++

Install the latest release with Python >= 3.10 and ``pip``:

.. code-block:: bash

    pip install earthkit-meteo

This minimal install does not include any optional dependencies.

Installing all optional packages
++++++++++++++++++++++++++++++++

To install **earthkit-meteo** with all optional packages:

.. code-block:: bash

    pip install "earthkit-meteo[all]"

.. note::

   Some shells (e.g. **zsh**) require quotes around the square brackets, as shown above.


Installing individual optional packages
+++++++++++++++++++++++++++++++++++++++

The following optional extras are available:

- ``lunar``: extra packages required for the :py:mod:`earthkit.meteo.lunar` submodule
- ``scores``: extra packages required for the :py:mod:`earthkit.meteo.score` submodule
- ``stats``: extra packages required for the :py:mod:`earthkit.meteo.stats` submodule

To install a single extra, for example ``scores``:

.. code-block:: bash

    pip install "earthkit-meteo[scores]"

Multiple extras can be combined in a single command:

.. code-block:: bash

    pip install "earthkit-meteo[scores,stats]"
