
Version 0.6 Updates
/////////////////////////


Version 0.6.0
===============


Refactored array backends
-------------------------

.. note::

    The array backend implementation in :xref:`earthkit-utils` has been refactored. The minimum required version of :xref:`earthkit-utils` is now ``0.2.0`` (:pr:`74`).

The public API of ``earthkit-meteo`` did not change.


Changes
-----------------------

- Uses :py:meth:`atan2` instead of :py:meth:`arctan2` internally for array-based computations (:pr:`63`)


Installation
-----------------------

- The minimum required version of Python is now 3.10
