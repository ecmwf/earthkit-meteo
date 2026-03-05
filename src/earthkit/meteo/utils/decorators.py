# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from abc import ABCMeta
from abc import abstractmethod
from functools import wraps
from importlib import import_module
from inspect import signature
from typing import Any

import xarray as xr
from earthkit.utils.array import array_namespace


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


class DataDispatcher(metaclass=ABCMeta):
    """
    A dispatcher class to route function calls based on input data types.
    """

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
        return _is_fieldlist(obj)

    def dispatch(self, func, module, *args, **kwargs):
        module = import_module(module + ".fieldlist")
        return getattr(module, func)(*args, **kwargs)


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
    The dispatch will attempt to route the call to the appropriate implementation based on the type of the specified argument.
    The implementations are assumed to live in submodules named after the data type (e.g., .xarray, .fieldlist, .array) with the same function name as the toplevel function.

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

    if isinstance(match, int):
        param_name = list(sig.parameters)[match]
    else:
        param_name = match

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
        from typing import get_args
        from typing import get_origin

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
    """
    Get dimension name from defaults if not provided.
    """
    if dim is not None:
        return dim
    for name in dim_names:
        if name in da.dims:
            return name
    return None


def xarray_ufunc_deprecated(**xarray_ufunc_kwargs):
    """
    Decorator for xarray wrappers that call the matching array implementation via xr.apply_ufunc.

    Parameters
    ----------
    xarray_ufunc_kwargs : dict, optional
        Default kwargs forwarded to xarray.apply_ufunc. Call-time overrides can be
        passed via the ``xarray_ufunc_kwargs`` kwarg on the wrapped function.
    """

    def decorator(func):
        module_name = func.__module__.replace(".xarray.", ".array.")

        @wraps(func)
        def wrapper(*args, **kwargs):
            call_ufunc_kwargs = kwargs.pop("xarray_ufunc_kwargs", None) or {}
            merged = {
                "dask": "parallelized",
                "keep_attrs": True,
            }
            if xarray_ufunc_kwargs:
                merged.update(xarray_ufunc_kwargs)
            merged.update(call_ufunc_kwargs)

            module = import_module(module_name)
            array_func = getattr(module, func.__name__)
            if "output_dtypes" not in merged:
                output_count = _infer_output_count(array_func)
                merged["output_dtypes"] = [float] * output_count

            if "output_core_dims" not in merged and len(merged["output_dtypes"]) > 1:
                output_core_dims = [args[0].dims for _ in merged["output_dtypes"]]
                merged["output_core_dims"] = output_core_dims

            if "input_core_dims" not in merged and len(merged["output_dtypes"]) > 1:
                input_core_dims = [x.dims for x in args]
                merged["input_core_dims"] = input_core_dims

            return xr.apply_ufunc(
                array_func,
                *args,
                kwargs=kwargs,
                **merged,
            )

        return wrapper

    return decorator


# def find_array_func():
#     module_name = func.__module__.replace(".xarray.", ".array.")
#     module = import_module(module_name)
#     array_func = getattr(module, func.__name__)


def xarray_ufunc(func, *args, **kwargs):
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
