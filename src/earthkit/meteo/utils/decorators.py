# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import wraps
from importlib import import_module
from inspect import signature
from itertools import repeat
from typing import TYPE_CHECKING, Any

from earthkit.utils.array import array_namespace

if TYPE_CHECKING:
    import xarray as xr


def _is_xarray(obj: Any) -> bool:
    from earthkit.meteo.utils import is_module_loaded

    if not is_module_loaded("xarray"):
        return False

    try:
        import xarray as xr

        return isinstance(obj, (xr.DataArray, xr.Dataset))
    except (ImportError, RuntimeError, SyntaxError):
        return False


def _is_fieldlist(obj: Any) -> bool:
    from earthkit.meteo.utils import is_module_loaded

    if not is_module_loaded("earthkit.data"):
        return False

    try:
        from earthkit.data import FieldList

        return isinstance(obj, FieldList)
    except ImportError:
        return False


def _is_field(obj: Any) -> bool:
    from earthkit.meteo.utils import is_module_loaded

    if not is_module_loaded("earthkit.data"):
        return False

    try:
        from earthkit.data import Field

        return isinstance(obj, Field)
    except ImportError:
        return False


class DataDispatcher(metaclass=ABCMeta):
    """A dispatcher class to route function calls based on input data types."""

    @staticmethod
    @abstractmethod
    def match(obj: Any) -> bool:
        pass

    @abstractmethod
    def dispatch(self, func: str, module: str, *args: Any, **kwargs: Any) -> Any:
        pass


class XArrayDispatcher(DataDispatcher):
    @staticmethod
    def match(obj: Any) -> bool:
        return _is_xarray(obj)

    def dispatch(self, func, module, *args, **kwargs):
        module = import_module(module + ".xarray")
        return getattr(module, func)(*args, **kwargs)


class FieldListDispatcher(DataDispatcher):
    @staticmethod
    def match(obj: Any) -> bool:
        return _is_fieldlist(obj) or _is_field(obj)

    def dispatch(self, func, module, *args, **kwargs):
        module = import_module(module + ".fieldlist")
        func = getattr(module, func, None)
        sig = signature(func)
        _kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return func(*args, **_kwargs)


class ArrayDispatcher(DataDispatcher):
    @staticmethod
    def match(obj: Any) -> bool:
        xp = array_namespace(obj)
        try:
            xp.asarray(obj)
            return True
        except Exception:
            return False

    def dispatch(self, func, module, *args, **kwargs):
        module = import_module(module + ".array")
        return getattr(module, func)(*args, **kwargs)


_DISPATCHERS = [XArrayDispatcher(), FieldListDispatcher(), ArrayDispatcher()]


def dispatch(func, match=0, xarray=True, fieldlist=True, array=False):
    """
    Decorator to dispatch function calls based on input data types.
    The dispatch will attempt to route the call to the appropriate
    implementation based on the type of the specified argument.
    The implementations are assumed to live in submodules named after the data
    type (e.g., .xarray, .fieldlist, .array) with the same function name as
    the toplevel function.

    Parameters
    ----------
    func: function
        The toplevel function to be decorated.
    match: int or str
        The index or name of the argument to check for dispatching. Default is 0 (the first argument).
    xarray: bool
        Whether to include the xarray dispatcher. Default is True.
    fieldlist: bool
        Whether to include the FieldList dispatcher. Default is True.
    array: bool
        Whether to include the array dispatcher. Default is False.

    Returns
    -------
    function
        The decorated function with dispatching capability.
    """
    DISPATCHERS = []
    if xarray:
        DISPATCHERS.append(_DISPATCHERS[0])
    if fieldlist:
        DISPATCHERS.append(_DISPATCHERS[1])
    if array:
        DISPATCHERS.append(_DISPATCHERS[2])

    sig = signature(func)

    params = list(sig.parameters)
    if isinstance(match, int):
        try:
            param_name = params[match]
        except IndexError as e:
            raise ValueError(
                f"'match' index {match} is invalid for function {func.__name__} with  {len(params)} arguments"
            ) from e
    elif isinstance(match, str):
        if match in params:
            param_name = match
        else:
            raise ValueError(f"'match' parameter name {match} is not in the function signature of {func.__name__}")
    else:
        raise TypeError(f"'match' must be an integer index or a string parameter name, got {type(match)}")

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        obj_to_check = bound_args.arguments[param_name]

        _module = ".".join(func.__module__.split(".")[:-1])
        for dispatcher in DISPATCHERS:
            if dispatcher.match(obj_to_check):
                return dispatcher.dispatch(func.__name__, _module, *args, **kwargs)
        raise TypeError(f"No matching dispatcher found for the input type: {type(obj_to_check)}")

    return wrapper


def _infer_output_count(func) -> int:
    try:
        import inspect
        from typing import get_args, get_origin

        annotation = inspect.signature(func).return_annotation
    except (ValueError, TypeError):
        return 1

    if annotation is inspect.Signature.empty:
        return 1

    origin = get_origin(annotation)
    if origin is tuple:
        args = get_args(annotation)
        if args and args[-1] is not Ellipsis:
            return len(args)
    return 1


def get_dim_from_defaults(da: xr.DataArray, dim: str | None, dim_names: tuple[str, ...]) -> str | None:
    """Get dimension name from defaults if not provided."""
    if dim is not None:
        return dim
    for name in dim_names:
        if name in da.dims:
            return name
    return None


def xarray_ufunc(func, *args, **kwargs):
    try:
        import xarray as xr
    except ImportError as e:
        raise RuntimeError("xarray dependency is required") from e

    xarray_ufunc_kwargs = kwargs.pop("xarray_ufunc_kwargs", None) or {}
    merged = {
        "dask": "parallelized",
        "keep_attrs": True,
    }
    if xarray_ufunc_kwargs:
        merged.update(xarray_ufunc_kwargs)

    if "output_dtypes" not in merged:
        output_count = _infer_output_count(func)
        merged["output_dtypes"] = [float] * output_count

    if "output_core_dims" not in merged and len(merged["output_dtypes"]) > 1:
        output_core_dims = [args[0].dims for _ in merged["output_dtypes"]]
        merged["output_core_dims"] = output_core_dims

    if "input_core_dims" not in merged and len(merged["output_dtypes"]) > 1:
        input_core_dims = [x.dims for x in args]
        merged["input_core_dims"] = input_core_dims

    return xr.apply_ufunc(
        func,
        *args,
        kwargs=kwargs,
        **merged,
    )


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
