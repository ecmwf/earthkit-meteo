.. _release-notes-1.0:

Version 1.0 Updates
///////////////////////


Version 1.0.0
===============


Deprecated features
------------------------
The following deprecations have been introduced in this release:

-  :ref:`deprecated-hybrid-pressure-at-model-levels`
-  :ref:`deprecated-hybrid-relative-geopotential-thickness`
-  :ref:`deprecated-hybrid-pressure-at-height-levels`

Please note that the import paths for the deprecated functions have changed. See the next section for details.


Changed import paths
------------------------
The deprecated functions listed above now have to be imported from the :py:mod:`earthkit.meteo.vertical.array` submodule. Previously they were available in the :py:mod:`earthkit.meteo.vertical` submodule.

Removed
----------------

The following functions have been removed:

- :py:func:`earthkit.meteo.thermo.kelvin_to_celsius`
- :py:func:`earthkit.meteo.thermo.celsius_to_kelvin`

They can be both replaced by using the newly added :py:attr:`~earthkit.meteo.constants.T_C2K` (273.15 K) constant.


High level interface
-----------------------
A unified high-level interface has been added across most modules, supporting array, Xarray and :xref:`earthkit-data` FieldList/Field input. The appropriate implementation is selected automatically based on the input type, so the same function names and call signatures work regardless of how the data is represented.

.. note::

    Not all functions support every input type yet, and there may be minor API differences between implementations. Refer to each function's documentation for details.

.. code-block:: python

    from earthkit.meteo import wind

    # For array-based inputs
    speed = wind.speed(u_array, v_array)

    # For Xarray-based inputs
    speed = wind.speed(u_xarray, v_xarray)

    # For FieldList-based inputs
    speed = wind.speed(u_fieldlist, v_fieldlist)


The actual implementations are now available in the ``array``, ``xarray`` and ``fieldlist``
submodules and can also be directly accessed bypassing the high level dispatch mechanism.

.. code-block:: python

    import earthkit.meteo.wind.array as array_wind
    import earthkit.meteo.wind.xarray as xarray_wind
    import earthkit.meteo.wind.fieldlist as fieldlist_wind

    # For array-based inputs
    speed = array_wind.speed(u_array, v_array)

    # For Xarray-based inputs
    speed = xarray_wind.speed(u_xarray, v_xarray)

    # For FieldList-based inputs
    speed = fieldlist_wind.speed(u_fieldlist, v_fieldlist)


Examples of using the high-level interface with various inputs can be found in the following notebook example:

  - :ref:`/tutorials/input/input_formats.ipynb`


New vertical methods
------------------------------------------------

The following coordinate computing functions have been added for array and FieldList inputs (see: :py:mod:`earthkit.meteo.vertical.array` and :py:mod:`earthkit.meteo.vertical.fieldlist`):

  - hybrid_level_parameters
  - pressure_on_hybrid_levels
  - relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta
  - relative_geopotential_thickness_on_hybrid_levels
  - geopotential_on_hybrid_levels
  - height_on_hybrid_levels

The following vertical interpolation functions have been added for array and FieldList inputs (see: :py:mod:`earthkit.meteo.vertical.array` and :py:mod:`earthkit.meteo.vertical.fieldlist`):

  - interpolate_hybrid_to_pressure_levels
  - interpolate_hybrid_to_height_levels
  - interpolate_pressure_to_height_levels
  - interpolate_monotonic

The following vertical interpolation functions have been added to the :py:mod:`earthkit.meteo.vertical.xarray` submodule for Xarray inputs:

  - interpolate_monotonic
  - interpolate_to_pressure_levels
  - interpolate_sleve_to_coord_levels
  - interpolate_sleve_to_theta_levels


See all notebook examples for vertical coordinate and interpolation functions in the :ref:`vertical-tutorials` tutorial section.


Regimes
------------------------

Added new submodule :py:mod:`earthkit.meteo.regimes` with classes to define a weather regime classification based on patterns and functions to project anomaly fields onto these patterns to compute a regime index, following the approach of `Michel and Rivière (2011) <https://doi.org/10.1175/2011JAS3635.1>`_.

See the notebook example: :ref:`/tutorials/regimes/seven_weather_regimes.ipynb`


Bootstrap utilities
---------------------------------------

Added a new bootstrapping helper module :py:mod:`earthkit.meteo.score.bootstrap`
(:pr:`54`) providing functions for statistical resampling and bootstrap
confidence interval estimation. Both array and xarray inputs are supported.

Key functions:

- :py:func:`~earthkit.meteo.score.bootstrap.resample` — draw bootstrap samples
  from a dataset, with optional replacement
- :py:func:`~earthkit.meteo.score.bootstrap.bootstrap` — compute a statistic
  over bootstrap samples

New features
-------------

- Added option "bolton43" to compute the equivalent potential temperature using equation (43) from [Bolton1980]_ (:pr:`128`). See:

  - :py:func:`~earthkit.meteo.thermo.ept_from_dewpoint`
  - :py:func:`~earthkit.meteo.thermo.ept_from_specific_humidity`


``kge`` exposed at top-level score module
-------------------------------------------

:py:func:`~earthkit.meteo.score.kge` (Kling–Gupta Efficiency) is now importable
directly from :py:mod:`earthkit.meteo.score` (:pr:`162`).
