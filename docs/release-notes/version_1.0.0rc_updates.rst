.. _release-notes-1.0.0rc:

Version 1.0.0 Release Candidate Updates
///////////////////////////////////////


Version 1.0.0rc0
==================

Deprecations
----------------

- :ref:`deprecated-hybrid-pressure-at-model-levels`
- :ref:`deprecated-hybrid-relative-geopotential-thickness`
- :ref:`deprecated-hybrid-pressure-at-height-levels`

Removed
----------------

The following functions have been removed:

- :py:func:`earthkit.meteo.thermo.kelvin_to_celsius`
- :py:func:`earthkit.meteo.thermo.celsius_to_kelvin`

They can be both replaced by using the newly added :py:attr:`~earthkit.meteo.constants.T_C2K` (273.15 K) constant.


High level interface
-----------------------
Added a high-level interface to most of the functions to support both array-based and Xarray-based inputs (:pr:`119`). This offers automatic dispatching to the appropriate implementation based on the input type. This allows users to use the same function names and parameters regardless of the input type, while still benefiting from the optimized implementations for each type.

.. code-block:: python

    from earthkit.meteo.wind import wind_speed

    # For array-based inputs
    speed = wind_speed(u_array, v_array)

    # For Xarray-based inputs
    speed = wind_speed(u_xarray, v_xarray)


The actual implementations are now available in the ``array`` and ``xarray`` submodules and can also be directly accessed bypassing the high level dispatch mechanism.

.. code-block:: python

    from earthkit.meteo.wind.array import wind_speed as array_wind_speed
    from earthkit.meteo.wind.xarray import wind_speed as xarray_wind_speed

    # For array-based inputs
    speed = array_wind_speed(u_array, v_array)

    # For Xarray-based inputs
    speed = xarray_wind_speed(u_xarray, v_xarray)



Array-based vertical coordinate functions
------------------------------------------------

Added the following array-based functions to  :py:mod:`earthkit.meteo.vertical.array` to compute vertical coordinate parameters:

- :py:func:`~earthkit.meteo.vertical.array.hybrid_level_parameters`
- :py:func:`~earthkit.meteo.vertical.array.pressure_on_hybrid_levels`
- :py:func:`~earthkit.meteo.vertical.array.relative_geopotential_thickness_on_hybrid_levels_from_alpha_delta`
- :py:func:`~earthkit.meteo.vertical.array.relative_geopotential_thickness_on_hybrid_levels`
- :py:func:`~earthkit.meteo.vertical.array.geopotential_on_hybrid_levels`
- :py:func:`~earthkit.meteo.vertical.array.height_on_hybrid_levels`

See the notebook examples:

- :ref:`/how-tos/hybrid_levels.ipynb`

Array-based vertical interpolations
------------------------------------------

Added the following array-based vertical interpolation functions to :py:mod:`earthkit.meteo.vertical.array`:

- :py:func:`~earthkit.meteo.vertical.array.interpolate_hybrid_to_pressure_levels`
- :py:func:`~earthkit.meteo.vertical.array.interpolate_hybrid_to_height_levels`
- :py:func:`~earthkit.meteo.vertical.array.interpolate_pressure_to_height_levels`
- :py:func:`~earthkit.meteo.vertical.array.interpolate_monotonic`

See the notebook examples:

- :ref:`/how-tos/interpolate_hybrid_to_hl.ipynb`
- :ref:`/how-tos/interpolate_hybrid_to_pl.ipynb`
- :ref:`/how-tos/interpolate_pl_to_hl.ipynb`
- :ref:`/how-tos/interpolate_pl_to_pl.ipynb`


Xarray-based vertical interpolation functions
----------------------------------------------

Added the following Xarray-based interpolation functions to :py:mod:`earthkit.meteo.vertical.xarray`:

- :py:func:`~earthkit.meteo.vertical.xarray.interpolate_monotonic`
- :py:func:`~earthkit.meteo.vertical.xarray.interpolate_to_pressure_levels`
- :py:func:`~earthkit.meteo.vertical.xarray.interpolate_sleve_to_coord_levels`
- :py:func:`~earthkit.meteo.vertical.xarray.interpolate_sleve_to_theta_levels`

Regimes
------------------------

Added new submodule :py:mod:`earthkit.meteo.regimes` with classes to define a weather regime classification based on patterns and functions to project anomaly fields onto these patterns to compute a regime index, following the approach of `Michel and Rivière (2011) <https://doi.org/10.1175/2011JAS3635.1>`_.

See the notebook example: :ref:`/how-tos/seven_weather_regimes.ipynb`
