Development
===========

Contributions
-------------

The code repository is hosted on `Github`_, testing, bug reports and contributions are highly welcomed and appreciated. Feel free to fork it and submit your PRs against the **develop** branch.

Development setup
-----------------------

First, clone the repository locally. You can use the following command:

.. code-block:: shell

   git clone --branch develop git@github.com:ecmwf/earthkit-meteo.git

(optional) If using conda, create a new environment and activate it:

.. code-block:: shell

    conda create -n earthkit-meteo python=3.12
    conda activate earthkit-meteo

Lastly, enter your git repository and run the following commands:

.. code-block:: shell

    pip install -e .[dev]
    pre-commit install

This setup enables the `pre-commit`_ hooks, performing a series of quality control checks on every commit. If any of these checks fails the commit will be rejected.

Run unit tests
---------------

To run the test suite, you can use the following command:

.. code-block:: shell

    pytest


Build documentation
-------------------

To build the documentation locally, please install the Python dependencies first:

.. code-block:: shell

    pip install -e .[docs]
    cd docs
    make html

To see the generated HTML documentation open the ``docs/_build/html/index.html`` file in your browser.


.. _`Github`: https://github.com/ecmwf/earthkit-meteo
.. _`pre-commit`: https://pre-commit.com/
