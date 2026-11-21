.. _fieldlist-result-metadata:

Setting metadata for FieldList results
======================================

This page is for developers writing or maintaining the FieldList implementations of earthkit-meteo
(``src/earthkit/meteo/*/fieldlist/*.py``). It explains how the metadata of the resulting Field or
FieldList is set. What this metadata means for users (GRIB encoding, synchronisation, limitations)
is described in :ref:`field-resulting-metadata`.

The parameter database
----------------------

Every parameter earthkit-meteo computes has a definition in a shared database that lives in
earthkit-utils: ``earthkit.utils.parameters.PARAMS``, backed by the ``_PARAMS`` dictionary. The
keys of a definition are:

- ``variable``: the value of ``parameter.variable``
- ``units``: the value of ``parameter.units``
- ``long_name``, ``standard_name``, ``cf_units``: used by the xarray results (see :ref:`xarray-result-metadata`)
- ``paramId`` and ``edition`` (optional): the ecCodes GRIB parameter ID and the GRIB editions supporting it

A lookup can also accept definitions provided by the caller (see ``parameter_definitions`` below),
but the goal is to store all parameters in earthkit-utils. New parameters should be added there.

fieldlist_ufunc
---------------

Most fieldlist functions only compute the output values from the values of the input fields. The
``fieldlist_ufunc`` helper (re-exported by ``earthkit.meteo.utils.decorators`` from earthkit-utils)
does this: it applies ``func`` to the ``values`` of the input Fields, one field (or one field per
position when several FieldLists are given) at a time, and builds the result with
:py:meth:`earthkit.data.core.field.Field.set`, setting the parameter metadata of the new fields:

.. code-block:: python

    from earthkit.meteo.utils.decorators import fieldlist_ufunc
    from earthkit.meteo.thermo import array

    def virtual_temperature(t, q):
        fieldlist_ufunc_kwargs = {"default_parameter": "virtual_temperature"}
        return fieldlist_ufunc(array.virtual_temperature, t, q, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs)

Keyword arguments other than ``fieldlist_ufunc_kwargs`` are passed on to ``func``. The result is a
Field when the inputs are Fields, otherwise a FieldList.

Choosing the output parameter
-----------------------------

The metadata set on the result is ``parameter.variable`` and ``parameter.units``, looked up in the
parameter database using the output parameter name. That name is determined from the
``fieldlist_ufunc_kwargs`` options below, the first match wins:

1. ``parameters``: a dict mapping the input field's ``parameter.variable`` to the output parameter name.
2. ``param_ids``: a dict mapping the input field's ``metadata.paramId`` (the ecCodes GRIB parameter ID) to the output parameter name.
3. ``default_parameter``: the parameter name used when neither mapping yields a result.

If no name can be determined, a ``ValueError`` is raised. The mappings make it possible to keep the
difference between similar parameters. This is the real code of
:func:`earthkit.meteo.thermo.fieldlist.dewpoint_from_relative_humidity`:

.. code-block:: python

    param_ids = {130: "dewpoint", 167: "2m_dewpoint"}
    parameters = {"t": "dewpoint", "2t": "2m_dewpoint"}

    fieldlist_ufunc_kwargs = {
        "param_ids": param_ids,
        "parameters": parameters,
        "default_parameter": "dewpoint",
    }
    return fieldlist_ufunc(
        array.dewpoint_from_relative_humidity, t, r, fieldlist_ufunc_kwargs=fieldlist_ufunc_kwargs
    )

Dewpoint computed from temperature (``parameter.variable`` "t" or ``paramId`` 130) results in the
"dewpoint" parameter, while computing it from 2 metre temperature ("2t" / 167) results in the
"2m_dewpoint" parameter. Functions where the output parameter does not depend on the input, such as
:func:`earthkit.meteo.wind.fieldlist.direction`, only set ``default_parameter``.

The chosen name must be defined in the parameter database (or in ``parameter_definitions``),
otherwise a ``KeyError`` is raised.

Setting the vertical level
--------------------------

The ``vertical`` option of ``fieldlist_ufunc_kwargs`` is passed unchanged to ``Field.set`` under the
"vertical" key. It is needed when the result is defined on a different level than the inputs, e.g.
in :func:`earthkit.meteo.vertical.fieldlist.extrapolate_temperature_to_pressure_levels_below_surface`:

.. code-block:: python

    fieldlist_ufunc_kwargs = {
        "default_parameter": "temperature",
        "vertical": {"level_type": "pressure", "level": target_p / 100.0},
    }

Note that earthkit-data stores pressure levels in hPa while ``target_p`` is given in Pa.

parameter_definitions
---------------------

Parameters that are not yet in the earthkit-utils database can be supplied per call with the
``parameter_definitions`` option. It has the same shape as the database: a dict keyed by parameter
name, whose values must define at least the ``variable``, ``units``, ``long_name``,
``standard_name`` and ``cf_units`` keys (``paramId`` and ``edition`` are optional). Caller-supplied
definitions are looked up before the built-in ones, so they override a built-in definition of the
same name for that call only. They are validated upfront, even when they end up not being used, and
are never modified:

.. code-block:: python

    fieldlist_ufunc_kwargs = {
        "default_parameter": "my_parameter",
        "parameter_definitions": {
            "my_parameter": {
                "variable": "mp",
                "units": "K",
                "long_name": "My parameter",
                "standard_name": None,
                "cf_units": None,
            }
        },
    }

This is meant as a development aid so that earthkit-meteo can work with parameters before they are
added to earthkit-utils. Parameters used in the long term should be added to the database.

Without fieldlist_ufunc
-----------------------

When a function cannot use ``fieldlist_ufunc`` (for instance because its values are not computed
from the values of the input fields), it can set the metadata itself.
``PARAMS.field_parameter_metadata(name)`` returns the ``parameter.variable``, ``parameter.units``
and ``parameter.standard_name`` of a parameter, ready to be merged into the dict given to
``Field.set``. This is how :mod:`earthkit.meteo.solar.fieldlist` works:

.. code-block:: python

    from earthkit.utils.parameters import PARAMS

    out_metadata = PARAMS.field_parameter_metadata("cos_solar_zenith_angle")

    result = []
    for field in data:
        lat, lon = field.geography.latlons()
        v = array.cos_solar_zenith_angle(date, lat, lon)
        result.append(field.set({"values": v, **out_metadata}))
    return FieldList.from_fields(result)

The interpolation helpers of :mod:`earthkit.meteo.vertical.fieldlist` use the same database to build
a FieldList from arrays and levels.

.. seealso::

    :ref:`xarray-result-metadata` and :ref:`field-resulting-metadata`
