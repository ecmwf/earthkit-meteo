# (C) Copyright 2026 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np
import pytest
from earthkit.utils.array import array_namespace
from earthkit.utils.array.testing import NAMESPACE_DEVICES

from earthkit.meteo import wind
from earthkit.meteo.utils.testing import NO_XARRAY


def _signature(obj):
    if isinstance(obj, tuple):
        return tuple(_signature(x) for x in obj)

    if hasattr(obj, "dims") and hasattr(obj, "shape"):
        return ("xarray", tuple(obj.dims), tuple(obj.shape))

    if hasattr(obj, "metadata") and hasattr(obj, "values") and hasattr(obj, "__len__"):
        return ("fieldlist", len(obj), tuple(obj.values.shape))

    xp = array_namespace(obj)
    arr = xp.asarray(obj)
    return ("array", tuple(arr.shape))


def _da(values):
    import xarray as xr

    return xr.DataArray(np.asarray(values))


def _case_array(xp, device):
    import earthkit.meteo.wind.array as impl

    u = xp.asarray([0, 1, -1, np.nan], device=device)
    v = xp.asarray([1, 1, -1, 1], device=device)
    speed = xp.asarray([1.0, 2.0, np.nan], device=device)
    direction = xp.asarray([180.0, 90.0, 1.0], device=device)
    direction_wr = xp.asarray([180.0, 90.0, 45.0], device=device)
    omega = xp.asarray([1.2, 21.3], device=device)
    temp = xp.asarray([285.6, 261.1], device=device)
    pressure = xp.asarray([1000, 850], device=device) * 100.0
    lat = xp.asarray([-20, 0, 50], device=device)
    speed_bins = xp.asarray([0.0, 1.0, 2.0, 4.0], device=device)

    return {
        "impl": impl,
        "ops": {
            "speed": ((u, v), {}),
            "direction": ((u, v), {}),
            "xy_to_polar": ((u, v), {}),
            "polar_to_xy": ((speed, direction), {}),
            "w_from_omega": ((omega, temp, pressure), {}),
            "coriolis": ((lat,), {}),
            "windrose": (
                (speed, direction_wr),
                {"sectors": 4, "speed_bins": speed_bins, "percent": False},
            ),
        },
    }


def _case_xarray():
    import earthkit.meteo.wind.xarray as impl

    u = _da([0, 1, -1, np.nan])
    v = _da([1, 1, -1, 1])
    speed = _da([1.0, 2.0, np.nan])
    direction = _da([180.0, 90.0, 1.0])
    direction_wr = _da([180.0, 90.0, 45.0])
    omega = _da([1.2, 21.3])
    temp = _da([285.6, 261.1])
    pressure = _da([1000, 850]) * 100.0
    lat = _da([-20, 0, 50])
    speed_bins = [0.0, 1.0, 2.0, 4.0]

    return {
        "impl": impl,
        "ops": {
            "speed": ((u, v), {}),
            "direction": ((u, v), {}),
            "xy_to_polar": ((u, v), {}),
            "polar_to_xy": ((speed, direction), {}),
            "w_from_omega": ((omega, temp, pressure), {}),
            "coriolis": ((lat,), {}),
            "windrose": (
                (speed, direction_wr),
                {"sectors": 4, "speed_bins": speed_bins, "percent": False},
            ),
        },
    }


BACKEND_PARAMS = [
    pytest.param(("array", xp, device), id=f"array-{xp._earthkit_array_namespace_name}-{device}")
    for xp, device in NAMESPACE_DEVICES
]
if not NO_XARRAY:
    BACKEND_PARAMS.append(pytest.param(("xarray", None, None), id="xarray"))


@pytest.fixture(params=BACKEND_PARAMS)
def backend_case(request):
    kind, xp, device = request.param
    if kind == "xarray":
        return _case_xarray()
    return _case_array(xp, device)


@pytest.mark.parametrize(
    "op_name",
    [
        "speed",
        "direction",
        "xy_to_polar",
        "polar_to_xy",
        "w_from_omega",
        "coriolis",
        "windrose",
    ],
)
def test_highlevel_compatible_with_backend_api(backend_case, op_name):
    impl = backend_case["impl"]
    args, kwargs = backend_case["ops"][op_name]

    got = getattr(wind, op_name)(*args, **kwargs)
    ref = getattr(impl, op_name)(*args, **kwargs)

    assert _signature(got) == _signature(ref)
