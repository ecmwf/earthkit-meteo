# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
from functools import wraps
from typing import Any
import xarray as xr

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
                f"{da.name} attribute '{key}'='{da.attrs.get(key)}' "
                f"does not match allowed '{val}'"
            )


def _update_metadata(da: xr.DataArray, allowed_attrs: dict) -> xr.DataArray:
    """
    Update the DataArray attributes to match CF metadata.
    """
    for key, val in allowed_attrs.items():
        val = list(val)[0] if isinstance(val, set) else val
        da.attrs[key] = val
    return da
