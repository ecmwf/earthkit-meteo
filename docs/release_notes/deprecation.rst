Deprecations
=============


.. _deprecated-0.7.0:

Version 0.7.0
-----------------

.. _deprecated-pressure-at-hybrid-levels:

:func:`pressure_at_hybrid_levels` is deprecated
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Its functionality is replaced by the more generic :func:`pressure_on_hybrid_levels`. The old method is still available for backward compatibility but will be removed in a future release.


.. list-table::
   :header-rows: 0

   * - Deprecated code
   * -

        .. literalinclude:: include/deprec_hybrid_pressure.py

   * - New code
   * -

        .. literalinclude:: include/migrated_hybrid_pressure.py


.. _deprecated-relative-geopotential-thickness:

:func:`relative_geopotential_thickness` is deprecated
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Its functionality is replaced by the more generic :func:`relative_geopotential_thickness_on_hybrid_levels`. The old method is still available for backward compatibility but will be removed in a future release.


.. list-table::
   :header-rows: 0

   * - Deprecated code
   * -

        .. literalinclude:: include/deprec_hybrid_geopotential_thickness.py

   * - New code
   * -

        .. literalinclude:: include/migrated_hybrid_geopotential_thickness.py
