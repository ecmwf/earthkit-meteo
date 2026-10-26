.. _field-resulting-metadata:

Setting metadata for computed Fields
=====================================

When earthkit-meteo computes a new Field or FieldList, the result's metadata is updated via :py:meth:`earthkit.data.core.field.Field.set`, setting e.g. the high-level ``parameter.variable`` and ``parameter.units`` keys (see :ref:`earthkit-data:parameter_component` for details). Currently, ``parameter.variable`` is set to the corresponding ecCodes GRIB ``shortName`` - regardless of whether the underlying data is actually stored in GRIB format.

In the example below, virtual temperature is computed from temperature and specific humidity. The resulting field has ``parameter.variable`` set to ``vtmp`` and ``parameter.units`` set to ``K``, matching the ecCodes GRIB parameter definition for virtual temperature.

.. code-block:: python

    from earthkit.meteo import thermo
    from earthkit.data import from_source

    # get temperature and specific humidity fields from GRIB files
    t = from_source("file_t.grib")[0]
    q = from_source("file_q.grib")[0]

    tv = thermo.virtual_temperature(t, q)

    print(tv.get("parameter.variable"))  # prints "vtmp"
    print(tv.get("parameter.units"))  # prints "K"


Parameters not defined in GRIB
------------------------------------------

Not all the computed parameters are supported by ecCodes (i.e. they do not have a corresponding GRIB ``paramId`` and ``shortName``). In that case, the ``parameter.variable`` and ``parameter.units`` are set to the corresponding values defined in an internal parameter registry.

The tables below list all the resulting parameters not supported by ecCodes.

.. list-table::
   :header-rows: 1
   :widths: 35 20 20

   * - Parameter description
     - parameter.variable
     - parameter.units
   * - ``Coriolis parameter``
     - ``fc``
     - 1/s
   * - ``Cos solar zenith angle integrated``
     - ``cossza_integrated``
     - 1
   * - ``Dewpoint``
     - ``td``
     - K
   * - ``Geometric height above sea``
     - ``h``
     - m
   * - ``Hybrid alpha``
     - ``hybrid_alpha``
     - 1
   * - ``Hybrid delta``
     - ``hybrid_delta``
     - 1
   * - ``LCL pressure``
     - ``p_lcl``
     - Pa
   * - ``LCL temperature``
     - ``t_lcl``
     - K
   * - ``Pressure full level``
     - ``pres``
     - Pa
   * - ``Pressure half level``
     - ``pres_half``
     - Pa
   * - ``Relative geopotential thickness``
     - ``rel_geopot_thick``
     - m²/s²
   * - ``Saturation mixing ratio``
     - ``ws``
     - kg/kg
   * - ``Saturation mixing ratio slope``
     - ``ws_slope``
     - kg/kg/K
   * - ``Saturation specific humidity``
     - ``sqw``
     - kg/kg
   * - ``Saturation specific humidity slope``
     - ``sqw_slope``
     - kg/kg/K
   * - ``Saturation vapour pressure slope``
     - ``swvp_slope``
     - Pa/K
   * - ``Specific gas constant``
     - ``R``
     - J/kg/K
   * - ``TOA incident solar radiation``
     - ``toa_incident_solar_radiation``
     - W/m²

Parameters encodable in GRIB edition 2 only
---------------------------------------------

There are some computed parameters that are only supported for GRIB edition 2 only. In that case the ecCodes ``shortName`` and ``units`` are set. As a consequence, it will be possible to encode the resulting field in GRIB edition 2, but not in GRIB edition 1.

The following parameters have an ecCodes ``paramId`` that is defined only for GRIB
edition 2 by ecCodes.

.. list-table::
   :header-rows: 1
   :widths: 35 20 15 15

   * - Parameter description
     - parameter.variable
     - parameter.units
     - ecCodes paramId
   * - ``10m wind direction``
     - ``10wdir``
     - degrees
     - 260260
   * - ``Geometric vertical velocity``
     - ``wz``
     - m/s
     - 260238
   * - ``Mass mixing ratio``
     - ``mass_mixrat``
     - kg/kg
     - 402000
   * - ``Saturation vapour pressure``
     - ``swvp``
     - Pa
     - 261021
   * - ``Vapour pressure``
     - ``vapp``
     - Pa
     - 260008
   * - ``Virtual potential temperature``
     - ``vptmp``
     - K
     - 3012
   * - ``Wet bulb potential temperature``
     - ``wbpt``
     - K
     - 261022
   * - ``Wet bulb temperature``
     - ``wbgt``
     - K
     - 261014


Using field metadata
-------------------------

If you only need to use the high-level (format agnostic) metadata on the resulting Fields/FieldLis the GRIB related limits do not apply. For example, you can plot the resulting fields or use them in further computations without any issues. However, if any point you need to access raw ecCodes GRIB metadata or encode the resulting fields in GRIB format, you need to consider the limitations described in next sections.

Accessing raw GRIB metadata
---------------------------------

When the input data is in GRIB format, the computed field's metadata will be out of sync with the underlying GRIB data (which is not updated).

.. note::

    This is done on purpose, avoiding unnecessary updates of the underlying GRIB data, which can be expensive or not even possible. The resulting field's high-level metadata is updated to reflect the computed parameter, but the underlying GRIB data is not updated.

If you do need to access the raw ecCodes GRIB metadata, the field has to be synchronized with the underlying GRIB data via :py:meth:`earthkit.data.core.field.Field.sync`.

.. code-block:: python

    from earthkit.meteo import thermo
    from earthkit.data import from_source

    # get temperature and specific humidity fields from GRIB files
    t = from_source("file_t.grib")[0]
    q = from_source("file_q.grib")[0]

    tv = thermo.virtual_temperature(t, q)

    # returns None, since the underlying GRIB data is not updated
    print(tv.get("metadata.shortName"))

    # Synchronize the field with the underlying GRIB data
    tv = tv.sync()

    # returns "vtmp", the shortName of the underlying GRIB data
    print(tv.get("metadata.shortName"))


When the resulting field is not encodable in GRIB, the synchronization will fail with an error message. E.g. if we compute the specific gas constant for moist air, the resulting field cannot be synchronized with the underlying GRIB data.

.. code-block:: python

    from earthkit.meteo import thermo
    from earthkit.data import from_source

    q = from_source("file_q.grib")[0]
    R = thermo.specific_gas_constant(q)

    # returns None, since the underlying GRIB data is not updated
    print(R.get("metadata.shortName"))

    # This will raise an error, since the parameter is not defined in GRIB
    R = R.sync()

The same is happening when the computed parameter is encodable in GRIB edition 2 only, but the underlying GRIB data is in edition 1. In that case, the synchronization will fail with an error message.

.. code-block:: python

    from earthkit.meteo import thermo
    from earthkit.data import from_source

    # specific humidity on pressure level in GRIB edition 1
    q = from_source("file_q.grib")[0]

    vp = thermo.vapour_pressure_from_specific_humidity(q)

    # returns None, since the underlying GRIB data is not updated
    print(vp.get("metadata.shortName"))

    # This will raise an error, since the parameter is encodable in GRIB edition 2 only
    vp = vp.sync()


Writing the results to GRIB
----------------------------------------

When the input data is in GRIB format, in most of the cases the computed field can be encoded in GRIB format as well.

.. code-block:: python

    from earthkit.meteo import thermo
    from earthkit.data import from_source

    t = from_source("file_t.grib")[0]
    q = from_source("file_q.grib")[0]

    tv = thermo.virtual_temperature(t, q)

    # Encode the resulting field in GRIB format
    tv.to_target("file", "tv.grib")

However, if the computed parameter is not defined in GRIB, the encoding will fail with an error message. E.g. if we compute the specific gas constant for moist air, the resulting field cannot be encoded in GRIB format.

.. code-block:: python

    from earthkit.meteo import thermo
    from earthkit.data import from_source

    q = from_source("file_q.grib")[0]
    R = thermo.specific_gas_constant(q)

    print(R.get("parameter.variable"))  # prints "R"
    print(R.get("parameter.units"))  # prints "J/kg/K"

    # This will raise an error, since the parameter is not defined in GRIB
    R.to_target("file", "R.grib")

Another problem arises when the computed parameter is encodable in GRIB edition 2 only. In that case, the encoding will fail if the user tries to encode the field in GRIB edition 1.

.. code-block:: python

    from earthkit.meteo import thermo
    from earthkit.data import from_source

    # specific humidity on pressure level in GRIB edition 1
    q = from_source("file_q.grib")[0]

    vp = thermo.vapour_pressure_from_specific_humidity(q)

    print(vp.get("parameter.variable"))  # prints "vapp"
    print(vp.get("parameter.units"))  # prints "Pa"

    # This will raise an error, since the parameter is encodable in GRIB edition 2 only
    vp.to_target("file", "vptmp.grib")

    # we need to explicitly use edition 2 to encode the field in GRIB format
    vp.to_target("file", "vptmp.grib", edition=2)
