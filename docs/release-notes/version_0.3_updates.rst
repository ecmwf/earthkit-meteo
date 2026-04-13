Version 0.3 Updates
/////////////////////////


Version 0.3.0
===============

- Removed the ``eps`` and ``out`` keywords arguments to control zero humidity in

  - :py:meth:`dewpoint_from_specific_humidity <earthkit.meteo.thermo.array.dewpoint_from_specific_humidity>`
  - :py:meth:`dewpoint_from_relative_humidity <earthkit.meteo.thermo.array.dewpoint_from_relative_humidity>`
  - :py:meth:`temperature_from_saturation_vapour_pressure <earthkit.meteo.thermo.array.temperature_from_saturation_vapour_pressure>`

  This was done to revert the changes added in version 0.2.0.
