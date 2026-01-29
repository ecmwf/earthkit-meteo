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
from typing import Any

import xarray as xr
import numpy as np
import numpy as np

# from earthkit.meteo.utils.metadata import metadata_handler
METADATA_DEFAULTS: dict[str, dict[str, object]] = {
    "mixing_ratio": {
        "standard_name": {"humidity_mixing_ratio"},
        "units": {"kg kg-1", "kg/kg"},
        "long_name": {
            "mixing ratio",
            "water vapor mixing ratio",
        },
    },
    "specific_humidity": {
        "standard_name": {"specific_humidity"},
        "units": {"kg kg-1", "kg/kg"},
        "long_name": {
            "specific humidity",
        },
    },
    "temperature": {
        "standard_name": {"air_temperature"},
        "units": {"K", "kelvin", "Celsius", "C"},
        "long_name": {
            "air temperature",
        },
    },
    "dewpoint": {
        "standard_name": {"dewpoint_temperature"},
        "units": {"K", "kelvin", "Celsius", "C"},
        "long_name": {
            "dewpoint temperature",
        },
    },
    "relative_humidity": {
        "standard_name": {"relative_humidity"},
        "units": {"%"},
        "long_name": {
            "relative humidity",
        },
    },
}


def metadata_handler(inputs: list[str] | dict[str, Any], outputs: list[str] | dict[str, Any]):
    """
    Decorator to check and update CF metadata on inputs and outputs of a function.

    - Inputs: validates that inputs conform to CF attributes.
    - Outputs: fills in or corrects metadata according to CF metadata dictionary.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # --- check inputs ---
            for var_name, arg in zip(inputs, args):
                if var_name not in METADATA_DEFAULTS:
                    continue
                allowed = METADATA_DEFAULTS[var_name]
                if isinstance(arg, xr.DataArray):
                    # validate units & attributes
                    _validate_and_warn(arg, allowed)

            # --- compute outputs ---
            result = func(*args, **kwargs)

            # normalize outputs to tuple for uniform handling
            outputs_list = result if isinstance(result, tuple) else (result,)
            updated_outputs = []

            # --- update outputs metadata ---
            for var_name, out in zip(outputs, outputs_list):
                if var_name not in METADATA_DEFAULTS:
                    updated_outputs.append(out)
                    continue
                allowed = METADATA_DEFAULTS[var_name]
                if isinstance(outputs, dict):
                    allowed |= outputs[var_name]
                if isinstance(out, xr.DataArray):
                    out = _update_metadata(out, allowed)
                updated_outputs.append(out)

            # return single object or tuple
            if isinstance(result, tuple):
                return tuple(updated_outputs)
            return updated_outputs[0]

        return wrapper

    return decorator


def _validate_and_warn(da: xr.DataArray, allowed_attrs: dict):
    """
    Validate the DataArray attributes. Raise a warning if they do not match.
    """
    import warnings

    # Check units
    if "units" in allowed_attrs and da.attrs.get("units") not in allowed_attrs["units"]:
        warnings.warn(
            f"{da.name} units '{da.attrs.get('units')}' do not match allowed '{allowed_attrs['units']}'"
        )

    # Check other attributes
    for key, val in allowed_attrs.items():
        if key == "units":
            continue
        if da.attrs.get(key).lower() not in val:
            warnings.warn(
                f"{da.name} attribute '{key}'='{da.attrs.get(key)}' " f"does not match allowed '{val}'"
            )


def _update_metadata(da: xr.DataArray, allowed_attrs: dict) -> xr.DataArray:
    """
    Update the DataArray attributes to match CF metadata.
    """
    for key, val in allowed_attrs.items():
        val = list(val)[0] if isinstance(val, set) else val
        da.attrs[key] = val
    return da


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


_DISPATCHERS = [XArrayDispatcher(), FieldListDispatcher()]


def dispatch(func):
    _module = ".".join(func.__module__.split(".")[:-1])
    # print(_module)

    @wraps(func)
    def inner(*args, **kwargs):
        for dispatcher in _DISPATCHERS:
            if dispatcher.match(args[0]):
                return dispatcher.dispatch(func.__name__, _module, *args, **kwargs)

    return inner


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


def xarray_ufunc(**xarray_ufunc_kwargs):
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
                input_core_dims = [x.dims for x in args]
                output_core_dims = [args[0].dims for _ in merged["output_dtypes"]]
                merged["input_core_dims"] = input_core_dims
                merged["output_core_dims"] = output_core_dims

            return xr.apply_ufunc(
                array_func,
                *args,
                kwargs=kwargs,
                **merged,
            )

        return wrapper

    return decorator
