Deprecations
=============


.. _deprecated-0.7.0:

Version 0.7.0
-----------------

.. _deprecated-hybrid-pressure-at-model-levels:

:func:`pressure_at_model_levels` is deprecated
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

It is replaced by the more generic :func:`pressure_on_hybrid_levels`. The old method is still available for backward compatibility but will be removed in a future release.


.. list-table::
   :header-rows: 0

   * - Deprecated code
   * -

        .. literalinclude:: include/deprec_hybrid_pressure.py

   * - New code
   * -

        .. literalinclude:: include/migrated_hybrid_pressure.py


.. _deprecated-hybrid-relative-geopotential-thickness:

:func:`relative_geopotential_thickness` is deprecated
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

It is replaced by the more generic :func:`relative_geopotential_thickness_on_hybrid_levels`. The old method is still available for backward compatibility but will be removed in a future release.


.. list-table::
   :header-rows: 0

   * - Deprecated code
   * -

        .. literalinclude:: include/deprec_hybrid_geopotential_thickness.py

   * - New code
   * -

        .. literalinclude:: include/migrated_hybrid_geopotential_thickness.py


.. _deprecated-hybrid-pressure-at-height-levels:

:func:`pressure_at_height_levels` is deprecated
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

It is replaced by the combined usage of :func:`pressure_on_hybrid_levels` and :func:`interpolate_hybrid_to_height_levels`. The old method is still available for backward compatibility but will be removed in a future release.


.. list-table::
   :header-rows: 0

   * - Deprecated code
   * -

        .. literalinclude:: include/deprec_hybrid_pressure_at_height_levels.py

   * - New code
   * -

        .. literalinclude:: include/migrated_hybrid_pressure_at_height_levels.py
