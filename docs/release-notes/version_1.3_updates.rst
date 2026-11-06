Version 1.3 Updates
/////////////////////////


Version 1.3.0
===============

New features
+++++++++++++++++++++++

- Added vertical extrapolation methods (:pr:`208`):

  - :py:func:`~earthkit.meteo.vertical.extrapolate_geopotential_to_pressure_levels_below_surface`
  - :py:func:`~earthkit.meteo.vertical.extrapolate_temperature_to_pressure_levels_below_surface`


Fixes
+++++++++++++++++++++++

- Fixed issue with using kg/kg coefficient in ``bolton43`` formula for computing equivalent potential temperature (:pr:`222`)
