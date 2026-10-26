# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from itertools import repeat

from earthkit.utils.decorators import dispatch as dispatch
from earthkit.utils.decorators import xarray_ufunc as xarray_ufunc


def get_dim_from_defaults(da, dim: str | None, dim_names: tuple[str, ...]) -> str | None:
    """Get dimension name from defaults if not provided."""
    if dim is not None:
        return dim
    for name in dim_names:
        if name in da.dims:
            return name
    return None


def field_ufunc(func, *args, **kwargs):
    """Apply a function to the values of earthkit.data Field or FieldList objects.

    Parameters
    ----------
    func: callable
        The function to apply to the values of the Field or FieldList objects.
    *args: tuple
        The Field or FieldList objects to which the function will be applied.
    **kwargs: dict
        Additional keyword arguments to pass to the function. The following special keyword arguments are recognized:
        - fieldlist_ufunc_kwargs: dict, optional
            A dictionary of keyword arguments to pass to the function when applied to FieldList objects.
            This can include 'variables', 'param_ids', 'default_variable'

            - 'variables': dict, optional
                A mapping of input field parameter.variable values to output parameter variable
                names. The output parameters names must be  defined in FIELD_PARAMS.
            - 'param_ids': dict, optional
                A mapping of input metadata.paramId values to output parameter variable names.
                The output parameters names must be defined in FIELD_PARAMS.
            - 'default_variable': str, optional
                The default parameter variable name to use if no mapping is found in
                'variables' or 'param_ids'. This must be defined in FIELD_PARAMS.

            The algorithm for determining the output parameter variable name is as follows:
            1. If 'variables' is provided, check if the first input field's parameter.variable
                is in the mapping. If so, use the corresponding output variable name.
            2. If 'param_ids' is provided, check if the first input field's metadata.paramId
                is in the mapping. If so, use the corresponding output variable name.
            3. If neither mapping yields a result, use 'default_variable' if provided.
            4. If no output parameter variable name can be determined, raise a ValueError.

            Once the output parameter variable name is determined, the corresponding metadata
            (parameter.variable and parameter.units) will be looked up in FIELD_PARAMS and set
            on the resulting Field.
    """
    import earthkit.data as ekd

    fieldlist_ufunc_kwargs = kwargs.pop("fieldlist_ufunc_kwargs", None) or {}

    fields = args
    field = fields[0]
    assert isinstance(field, ekd.Field), "field_ufunc first argument must be a Field"
    v = func(*(field.values if isinstance(field, ekd.Field) else field for field in fields), **kwargs)

    # determine the metadata to set on the resulting Field
    variables = fieldlist_ufunc_kwargs.get("variables", {})
    param_ids = fieldlist_ufunc_kwargs.get("param_ids", {})
    default = fieldlist_ufunc_kwargs.get("default_variable")

    name = None
    if variables:
        var_in = field.get("parameter.variable", default=None)
        if var_in is not None:
            name = variables.get(var_in)

    if name is None and param_ids:
        param_id_in = field.get("metadata.paramId", default=None)
        if param_id_in is not None:
            name = param_ids.get(param_id_in)

    if name is None:
        name = default

    if name is None:
        raise ValueError(
            "Could not determine parameter name for the resulting Field. Please provide "
            "a 'default_variable' in 'fieldlist_ufunc_kwargs'."
        )

    # look up the parameter metadata from FIELD_PARAMS
    from earthkit.meteo.utils.param import FIELD_PARAMS

    param_item = FIELD_PARAMS.get(name)

    if param_item is None:
        raise ValueError(f"Unknown parameter '{name}' specified in fieldlist_ufunc_kwargs")
    parameter_kwargs = {"parameter.variable": param_item["variable"], "parameter.units": param_item["units"]}
    result = field.set({"values": v, **parameter_kwargs})

    return result


def fieldlist_ufunc(func, *args, **kwargs):
    import earthkit.data as ekd

    if args:
        if isinstance(args[0], ekd.Field):
            return field_ufunc(func, *args, **kwargs)
        elif not (isinstance(args[0], ekd.FieldList)):
            raise TypeError(
                "fieldlist_ufunc arguments must be Field or FieldList instances. Found unsupported type: "
                + str(type(args[0]))
                + " in args"
            )
    else:
        raise ValueError("fieldlist_ufunc requires at least one argument")

    # an argument that is None is replaced with an infinite repeat of None to allow zipping without worrying
    # about lengths
    safe_args = [arg if arg is not None else repeat(None) for arg in args]

    result = []
    for fields in zip(*safe_args):
        result.append(
            field_ufunc(
                func,
                *fields,
                **kwargs,
            )
        )

    return ekd.FieldList.from_fields(result)
