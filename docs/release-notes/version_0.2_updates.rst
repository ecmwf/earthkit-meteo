Version 0.2 Updates
/////////////////////////


Version 0.2.0
===============

New features
+++++++++++++++

- Added the ``eps`` and ``out`` keywords arguments to control zero humidity in the following methods (:pr:`23`):

  - :py:meth:`dewpoint_from_specific_humidity <earthkit.meteo.thermo.array.dewpoint_from_specific_humidity>`
  - :py:meth:`dewpoint_from_relative_humidity <earthkit.meteo.thermo.array.dewpoint_from_relative_humidity>`
  - :py:meth:`temperature_from_saturation_vapour_pressure <earthkit.meteo.thermo.array.temperature_from_saturation_vapour_pressure>`
