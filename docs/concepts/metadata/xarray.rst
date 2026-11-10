.. _xarray-result-metadata:

Setting metadata for xarray results
===================================

This page is for developers writing or maintaining the xarray implementations of earthkit-meteo
(``src/earthkit/meteo/*/xarray/*.py``). It explains how the name and the attributes of the
resulting DataArray(s) are set.

The parameter database
----------------------

The parameter definitions live in the same shared database as for the FieldList results:
``earthkit.utils.parameters.PARAMS`` in earthkit-utils, backed by the ``_PARAMS`` dictionary. The
xarray metadata uses the ``variable`` key (the resulting DataArray name), plus ``long_name``,
``standard_name`` and ``units``/``cf_units`` for the attributes. The goal is to store all parameters
in earthkit-utils; new parameters should be added there. Temporary, per-call definitions can be
given with ``parameter_definitions`` (see below). See :ref:`fieldlist-result-metadata` for the
FieldList side.

xarray_ufunc
------------

xarray functions wrap :func:`xarray.apply_ufunc` with the ``xarray_ufunc`` helper (re-exported by
``earthkit.meteo.utils.decorators`` from earthkit-utils). It applies ``func`` to the inputs and
sets the name and the attributes of the result, described by the ``xarray_ufunc_kwargs`` options:

.. code-block:: python

    from earthkit.meteo.utils.decorators import xarray_ufunc
    from earthkit.meteo.wind import array

    def speed(u, v):
        return xarray_ufunc(array.speed, u, v, xarray_ufunc_kwargs={"parameters": "wind_speed"})

Keyword arguments other than ``xarray_ufunc_kwargs`` are passed on to ``func``.

"parameters"
------------

The ``parameters`` option names the output parameter(s) in the database. The result is renamed to
the ``variable`` of the parameter and its ``long_name``, ``standard_name`` and ``units`` attributes
are set. The units attribute uses the CF canonical units (``cf_units``) when defined, otherwise the
plain ``units``:

.. code-block:: python

    >>> s = speed(u, v)
    >>> s.name
    'ws'
    >>> s.attrs["long_name"]
    'Wind speed'
    >>> s.attrs["standard_name"]
    'wind_speed'
    >>> s.attrs["units"]
    'm s-1'

The metadata keys that are not defined by the parameter are removed from the result instead of
being inherited from the input. This keeps stale input attributes, e.g. an old ``units``, out of the
result even though ``keep_attrs=True`` is the default. Other attributes (``author``, ``grid``, ...)
are inherited.

For multi-output functions ``parameters`` is a list or tuple with one item per output; an empty
item (``""`` or ``None``) leaves that output untouched. This is the real code of
:func:`earthkit.meteo.thermo.xarray.lcl`, which returns two DataArrays named "t_lcl" and "p_lcl":

.. code-block:: python

    return xarray_ufunc(
        array.lcl,
        t,
        td,
        p,
        method=method,
        xarray_ufunc_kwargs={
            "parameters": ("lcl_temperature", "lcl_pressure"),
            "xarray_apply_ufunc_kwargs": {"output_dtypes": [float, float]},
        },
    )

"attrs"
-------

The ``attrs`` option adds attributes to the result or overrides the ones coming from the parameter.
It is a single dict applied to all outputs or, for multi-output functions, a list/tuple with one
item (a dict or None) per output. Values set to ``None`` remove the attribute. ``attrs`` requires
``parameters``, otherwise a ``ValueError`` is raised. From
:func:`earthkit.meteo.thermo.xarray.saturation_vapour_pressure`:

.. code-block:: python

    xarray_ufunc_kwargs={
        "parameters": "saturation_vapour_pressure",
        "attrs": {"comment": f"phase: {phase} method: {method}"},
    }

"parameter_definitions"
-----------------------

Parameters that are not yet in the earthkit-utils database can be supplied per call with
``parameter_definitions``, a dict with the same shape as ``_PARAMS``: the required keys are
``variable``, ``units``, ``long_name``, ``standard_name`` and ``cf_units``; ``paramId`` and
``edition`` are optional. Caller-supplied definitions take precedence over the built-in ones for
that call only and are validated even when they are not used:

.. code-block:: python

    xarray_ufunc_kwargs={
        "parameters": "my_parameter",
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

The resulting DataArray is named "mp" and has ``long_name`` "My parameter". This option exists so
that earthkit-meteo can use parameters before they are added to earthkit-utils; eventually all
parameters should live in the database.

Options passed to apply_ufunc
-----------------------------

The nested ``xarray_apply_ufunc_kwargs`` option is a dict of keyword arguments forwarded to
:func:`xarray.apply_ufunc`. It overrides the defaults, which are ``dask="parallelized"`` and
``keep_attrs=True``. It is needed when the function reduces or reshapes dimensions, e.g. the real
code of :func:`earthkit.meteo.extreme.xarray.sot`:

.. code-block:: python

    xarray_ufunc(
        array.sot,
        clim,
        ens,
        ...,
        xarray_ufunc_kwargs={
            "xarray_apply_ufunc_kwargs": {
                "input_core_dims": [[clim_dim], [ens_dim]],
                "output_core_dims": [[]],
            },
        },
    )

For backward compatibility, keys in ``xarray_ufunc_kwargs`` that are not one of the options
described here (``"parameters"``, ``"attrs"``, ``"parameter_definitions"`` and
``"xarray_apply_ufunc_kwargs"``) are also forwarded to :func:`xarray.apply_ufunc`, as if they had
been given in ``xarray_apply_ufunc_kwargs``. When the same key is present in both, the nested
``xarray_apply_ufunc_kwargs`` wins. Prefer the nested form in new code.

Defaults and inference
----------------------

When ``xarray_apply_ufunc_kwargs`` does not provide them:

- ``output_dtypes`` is inferred as one ``float`` per output. The number of outputs is taken from the
  return annotation of ``func``: a fixed-length ``tuple[...]`` annotation means as many outputs.
- For multi-output functions, ``output_core_dims`` is set to the dimensions of the first input and
  ``input_core_dims`` to the dimensions of every input.

So element-wise multi-output functions need no apply_ufunc options at all. Single-output
element-wise functions reduce to a plain call like the ``speed`` example above.

.. seealso::

    :ref:`fieldlist-result-metadata` and :ref:`field-resulting-metadata`
