Version 1.2 Updates
/////////////////////////

Version 1.2.0
===============

New features
+++++++++++++++++++++++

- Added the ``perc_tail`` keyword argument to :py:mod:`earthkit.meteo.extreme.sot` to include the percentile tail in the computation (:pr:`190`)
- Added a new computation method based on the [Huang_] formula to :py:func:`~earthkit.meteo.thermo.saturation_vapour_pressure`. The new method can be selected using the ``method`` keyword argument, with possible values:

    "ifs" (default): the current IFS formulation, results are unchanged;
    "huang": the formulas of [Huang_]
