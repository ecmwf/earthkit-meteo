Version 1.2 Updates
/////////////////////////


Version 1.2.1
===============

Fixes
+++++++++++++++++++++++

- Fixed :py:func:`~earthkit.meteo.wind.windrose` generating truncated direction bin edges when the number of sectors did not divide evenly into 360 degrees (:pr:`210`)
- Fixed the name of the optional extra required by the scoring functions in the error message raised when it is missing, changing it from ``score`` to ``scores`` (:pr:`215`)
- Fixed :py:func:`~earthkit.meteo.solar.cos_solar_zenith_angle` to use the fractional hour (including minutes and seconds) when computing the solar hour angle, instead of just the hour (:pr:`211`)
- Fixed :py:func:`~earthkit.meteo.thermo.wet_bulb_temperature_from_dewpoint` raising an error when called with 2D input and ``t_method="bisect"`` (:pr:`216`)



Version 1.2.0
===============

New features
+++++++++++++++++++++++

- Added the ``perc_tail`` keyword argument to :py:mod:`earthkit.meteo.extreme.sot` to include the percentile tail in the computation (:pr:`190`)
- Added a new computation method based on the [Huang2018]_ formula to :py:func:`~earthkit.meteo.thermo.saturation_vapour_pressure` (:pr:`204`). The new method can be selected using the ``method`` keyword argument, with possible values:

  - "ifs" (default): the current IFS formulation, results are unchanged
  - "huang": the formulas of [Huang2018]_

  The following functions are based on :py:func:`~earthkit.meteo.thermo.saturation_vapour_pressure` and also now have the ``method`` keyword argument:

  - :py:func:`~earthkit.meteo.thermo.saturation_vapour_pressure_slope`
  - :py:func:`~earthkit.meteo.thermo.saturation_mixing_ratio`
  - :py:func:`~earthkit.meteo.thermo.saturation_specific_humidity`
  - :py:func:`~earthkit.meteo.thermo.saturation_mixing_ratio_slope`
  - :py:func:`~earthkit.meteo.thermo.saturation_specific_humidity_slope`
