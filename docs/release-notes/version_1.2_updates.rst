Version 1.2 Updates
/////////////////////////

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
