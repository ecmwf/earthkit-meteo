
Version 0.5 Updates
/////////////////////////


Version 0.5.0
===============

New features
-----------------------

- Introduces the ``from_zero`` parameter to :py:meth:`earthkit.meteo.extreme.array.cpf`, default to False (previous behaviour corresponds to True), ignoring the lower half of the distribution. Also adds an extra check for "reverse crossings", i.e. when the forecast CDF starts above the climate distribution. (:pr:`67`)


Changes
-----------------------
- Uses :py:meth:`acos` instead of :py:meth:`arccos` internally for array-based computations (:pr:`56`)
