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
    import earthkit.data as ekd

    fieldlist_ufunc_kwargs = kwargs.pop("fieldlist_ufunc_kwargs", None) or {}
    variables = fieldlist_ufunc_kwargs.get("variables", {})
    param_ids = fieldlist_ufunc_kwargs.get("param_ids", {})
    default = fieldlist_ufunc_kwargs.get("default")
    unit = fieldlist_ufunc_kwargs.get("param_unit")

    fields = args
    u0 = fields[0]
    assert isinstance(u0, ekd.Field), "field_ufunc first argument must be a Field"
    v = func(*(field.values if isinstance(field, ekd.Field) else field for field in fields), **kwargs)

    name = None
    var_u = u0.get("parameter.variable", default=None)
    if var_u is not None:
        name = variables.get(var_u)
    else:
        var_u = u0.get("metadata.paramId", default=None)
        if var_u is not None:
            name = param_ids.get(var_u)
        else:
            var_u = "unknown"

    if default is None:
        default = var_u

    if name is None:
        name = default

    if unit is None:
        unit = u0.get("parameter.units")

    result = u0.set({"values": v, "parameter.variable": name, "parameter.units": unit})

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
